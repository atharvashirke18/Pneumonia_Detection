"""
tests/test_data_pipeline.py
Automated test suite for configuration, ManifestDataset, and PneumoniaDataModule.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import csv
import torch
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, ConfigError
from src.data.dataset import ManifestDataset
from src.data.datamodule import PneumoniaDataModule, get_transforms


class TestConfig(unittest.TestCase):
    """Test configuration loading, validation, and error handling."""

    def test_load_valid_config(self):
        cfg = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
        self.assertIn("project", cfg)
        self.assertIn("dataset", cfg)
        self.assertEqual(cfg["project"]["random_seed"], 42)
        self.assertEqual(cfg["preprocessing"]["image_size"], [224, 224])
        self.assertEqual(cfg["dataset"]["classes"], {"NORMAL": 0, "PNEUMONIA": 1})
        self.assertTrue(Path(cfg["dataset"]["data_root_resolved"]).exists())

    def test_missing_config_file(self):
        with self.assertRaises(ConfigError):
            load_config("configs/non_existent_config.yaml", base_dir=PROJECT_ROOT)

    def test_invalid_config_schema(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("project:\n  name: 'test'\n")
            temp_path = f.name

        try:
            with self.assertRaises(ConfigError):
                load_config(temp_path, base_dir=PROJECT_ROOT)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestManifestDatasetSynthetic(unittest.TestCase):
    """Test ManifestDataset using synthetic image fixtures."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create dummy images: RGB and Grayscale
        self.img_rgb = self.temp_path / "rgb_sample.jpeg"
        Image.new("RGB", (100, 100), color=(255, 0, 0)).save(self.img_rgb)

        self.img_gray = self.temp_path / "gray_sample.jpeg"
        Image.new("L", (80, 80), color=128).save(self.img_gray)

        # Create valid synthetic manifest
        self.manifest_file = self.temp_path / "test_manifest.csv"
        with open(self.manifest_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["assigned_split", "official_split", "label", "label_int", "patient_token", "filename", "rel_path", "abs_path"]
            )
            writer.writeheader()
            writer.writerow({
                "assigned_split": "train",
                "official_split": "train",
                "label": "NORMAL",
                "label_int": "0",
                "patient_token": "",
                "filename": self.img_gray.name,
                "rel_path": self.img_gray.name,
                "abs_path": str(self.img_gray),
            })
            writer.writerow({
                "assigned_split": "train",
                "official_split": "train",
                "label": "PNEUMONIA",
                "label_int": "1",
                "patient_token": "person101",
                "filename": self.img_rgb.name,
                "rel_path": self.img_rgb.name,
                "abs_path": str(self.img_rgb),
            })

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_synthetic_loading_and_conversion(self):
        transform = get_transforms(image_size=(224, 224), is_training=False)
        ds = ManifestDataset(
            manifest_path=self.manifest_file,
            data_root=self.temp_path,
            expected_split="train",
            transform=transform
        )
        self.assertEqual(len(ds), 2)

        # Check grayscale image converted to 3-channel RGB tensor
        img0, label0 = ds[0]
        self.assertEqual(img0.shape, torch.Size([3, 224, 224]))
        self.assertEqual(label0.item(), 0.0)

        # Check RGB image converted to tensor
        img1, label1 = ds[1]
        self.assertEqual(img1.shape, torch.Size([3, 224, 224]))
        self.assertEqual(label1.item(), 1.0)

    def test_rejection_of_unknown_class(self):
        bad_manifest = self.temp_path / "bad_class.csv"
        with open(bad_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["assigned_split", "label", "label_int", "filename", "rel_path", "abs_path"]
            )
            writer.writeheader()
            writer.writerow({
                "assigned_split": "train",
                "label": "COVID19",
                "label_int": "2",
                "filename": self.img_rgb.name,
                "rel_path": self.img_rgb.name,
                "abs_path": str(self.img_rgb),
            })

        with self.assertRaises(ValueError):
            ManifestDataset(manifest_path=bad_manifest, data_root=self.temp_path)

    def test_rejection_of_missing_file(self):
        missing_manifest = self.temp_path / "missing_file.csv"
        with open(missing_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["assigned_split", "label", "label_int", "filename", "rel_path", "abs_path"]
            )
            writer.writeheader()
            writer.writerow({
                "assigned_split": "train",
                "label": "NORMAL",
                "label_int": "0",
                "filename": "ghost.jpeg",
                "rel_path": "ghost.jpeg",
                "abs_path": str(self.temp_path / "ghost.jpeg"),
            })

        with self.assertRaises(FileNotFoundError):
            ManifestDataset(manifest_path=missing_manifest, data_root=self.temp_path)


class TestRealManifestsAndDataModule(unittest.TestCase):
    """Smoke and integrity tests against the audited manifests and DataModule."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
        cls.dm = PneumoniaDataModule(cls.cfg)
        cls.dm.setup()

    def test_manifest_counts(self):
        self.assertEqual(len(self.dm.train_dataset), 4183)
        self.assertEqual(len(self.dm.val_dataset), 518)
        self.assertEqual(len(self.dm.cal_dataset), 489)
        self.assertEqual(len(self.dm.test_dataset), 624)

        train_counts = self.dm.train_dataset.get_class_counts()
        self.assertEqual(train_counts["NORMAL"], 1072)
        self.assertEqual(train_counts["PNEUMONIA"], 3111)

        val_counts = self.dm.val_dataset.get_class_counts()
        self.assertEqual(val_counts["NORMAL"], 134)
        self.assertEqual(val_counts["PNEUMONIA"], 384)

        cal_counts = self.dm.cal_dataset.get_class_counts()
        self.assertEqual(cal_counts["NORMAL"], 134)
        self.assertEqual(cal_counts["PNEUMONIA"], 355)

        test_counts = self.dm.test_dataset.get_class_counts()
        self.assertEqual(test_counts["NORMAL"], 234)
        self.assertEqual(test_counts["PNEUMONIA"], 390)

    def test_patient_token_independence(self):
        train_tokens = set(self.dm.train_dataset.get_patient_tokens())
        val_tokens = set(self.dm.val_dataset.get_patient_tokens())
        cal_tokens = set(self.dm.cal_dataset.get_patient_tokens())

        self.assertEqual(len(train_tokens & val_tokens), 0, "Train and Val share patient tokens!")
        self.assertEqual(len(train_tokens & cal_tokens), 0, "Train and Cal share patient tokens!")
        self.assertEqual(len(val_tokens & cal_tokens), 0, "Val and Cal share patient tokens!")

    def test_train_dataloader_smoke_and_cuda(self):
        train_loader = self.dm.train_dataloader()
        batch_images, batch_labels = next(iter(train_loader))

        self.assertEqual(batch_images.shape, torch.Size([32, 3, 224, 224]))
        self.assertEqual(batch_labels.shape, torch.Size([32]))
        self.assertTrue(torch.all((batch_labels == 0.0) | (batch_labels == 1.0)))

        # Test CUDA transfer if available
        if torch.cuda.is_available():
            batch_images_cuda = batch_images.cuda()
            batch_labels_cuda = batch_labels.cuda()
            self.assertEqual(batch_images_cuda.device.type, "cuda")
            self.assertEqual(batch_labels_cuda.device.type, "cuda")

    def test_val_dataloader_smoke(self):
        val_loader = self.dm.val_dataloader()
        batch_images, batch_labels = next(iter(val_loader))

        self.assertEqual(batch_images.shape, torch.Size([32, 3, 224, 224]))
        self.assertEqual(batch_labels.shape, torch.Size([32]))

    def test_augmentation_vs_eval_deterministic(self):
        # Val dataset items should be identically transformed across fetches
        img_a, _ = self.dm.val_dataset[0]
        img_b, _ = self.dm.val_dataset[0]
        self.assertTrue(torch.allclose(img_a, img_b), "Evaluation transform is not deterministic!")


if __name__ == "__main__":
    unittest.main()
