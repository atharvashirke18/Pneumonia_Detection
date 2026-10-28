"""
src/config.py
Configuration loader and validator for the Pneumonia Detection Project.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

class ConfigError(ValueError):
    """Raised when configuration values are missing, invalid, or inconsistent."""
    pass

def load_config(config_path: str | Path, base_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Load and validate YAML configuration file.
    
    Args:
        config_path: Path to YAML file. Can be relative to base_dir or absolute.
        base_dir: Project root directory. Defaults to the directory containing this repo.
        
    Returns:
        Validated configuration dictionary with resolved paths.
    """
    if base_dir is None:
        # Default to parent directory of src/ (the repo root)
        base_dir = Path(__file__).resolve().parent.parent
    else:
        base_dir = Path(base_dir).resolve()
        
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = (base_dir / cfg_file).resolve()
        
    if not cfg_file.exists():
        raise ConfigError(f"Configuration file not found: {cfg_file}")
        
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Failed to parse YAML configuration at {cfg_file}: {e}")
        
    if not isinstance(cfg, dict):
        raise ConfigError(f"Configuration root must be a mapping, got {type(cfg).__name__}")
        
    _validate_and_resolve_config(cfg, base_dir)
    return cfg


def _validate_and_resolve_config(cfg: Dict[str, Any], base_dir: Path) -> None:
    """Validate structure and resolve relative paths against base_dir."""
    required_sections = ["project", "dataset", "preprocessing", "dataloader", "system", "model"]
    for sec in required_sections:
        if sec not in cfg:
            raise ConfigError(f"Missing required configuration section: '{sec}'")
            
    # 1. Project section
    proj = cfg["project"]
    if "random_seed" not in proj or not isinstance(proj["random_seed"], int):
        raise ConfigError("project.random_seed must be an integer")
        
    # 2. Dataset section
    ds = cfg["dataset"]
    if "data_root" not in ds:
        raise ConfigError("dataset.data_root is required")
    data_root = Path(ds["data_root"])
    if not data_root.is_absolute():
        data_root = (base_dir / data_root).resolve()
    if not data_root.exists() or not data_root.is_dir():
        raise ConfigError(f"dataset.data_root does not exist: {data_root}")
    ds["data_root_resolved"] = str(data_root)
    
    if "manifests_dir" not in ds:
        raise ConfigError("dataset.manifests_dir is required")
    manifests_dir = Path(ds["manifests_dir"])
    if not manifests_dir.is_absolute():
        manifests_dir = (base_dir / manifests_dir).resolve()
    if not manifests_dir.exists() or not manifests_dir.is_dir():
        raise ConfigError(f"dataset.manifests_dir does not exist: {manifests_dir}")
    ds["manifests_dir_resolved"] = str(manifests_dir)
    
    # Manifest files
    m_files = ds.get("manifest_files", {})
    required_splits = ["train", "val", "test"]
    for s in required_splits:
        if s not in m_files:
            raise ConfigError(f"dataset.manifest_files must define '{s}'")
        m_path = manifests_dir / m_files[s]
        if not m_path.exists():
            raise ConfigError(f"Required manifest file for split '{s}' not found: {m_path}")
            
    # Resolve all manifest paths into an accessible dictionary
    resolved_manifests = {}
    for split_name, fname in m_files.items():
        resolved_manifests[split_name] = str((manifests_dir / fname).resolve())
    ds["resolved_manifest_paths"] = resolved_manifests
    
    # Classes mapping
    classes = ds.get("classes", {})
    if not classes or not isinstance(classes, dict):
        raise ConfigError("dataset.classes must be a non-empty mapping (e.g. NORMAL: 0, PNEUMONIA: 1)")
    if set(classes.keys()) != {"NORMAL", "PNEUMONIA"}:
        raise ConfigError(f"dataset.classes must contain exact classes {{'NORMAL', 'PNEUMONIA'}}, got {set(classes.keys())}")
    if set(classes.values()) != {0, 1}:
        raise ConfigError(f"dataset.classes values must map to 0 and 1, got {set(classes.values())}")
        
    # 3. Preprocessing
    prep = cfg["preprocessing"]
    img_size = prep.get("image_size")
    if not (isinstance(img_size, (list, tuple)) and len(img_size) == 2 and all(isinstance(x, int) and x > 0 for x in img_size)):
        raise ConfigError(f"preprocessing.image_size must be a pair of positive integers [H, W], got {img_size}")
        
    norm = prep.get("normalization", {})
    if "mean" not in norm or "std" not in norm:
        raise ConfigError("preprocessing.normalization must define 'mean' and 'std'")
    if len(norm["mean"]) != 3 or len(norm["std"]) != 3:
        raise ConfigError("preprocessing.normalization mean and std must have 3 channels")
        
    # 4. DataLoader
    dl = cfg["dataloader"]
    if not isinstance(dl.get("batch_size"), int) or dl["batch_size"] <= 0:
        raise ConfigError(f"dataloader.batch_size must be a positive integer, got {dl.get('batch_size')}")
    if not isinstance(dl.get("num_workers"), int) or dl["num_workers"] < 0:
        raise ConfigError(f"dataloader.num_workers must be a non-negative integer, got {dl.get('num_workers')}")
        
    # 5. System
    sys_cfg = cfg["system"]
    if sys_cfg.get("device") not in ["cuda", "cpu"]:
        raise ConfigError(f"system.device must be 'cuda' or 'cpu', got {sys_cfg.get('device')}")
    if sys_cfg.get("mixed_precision") not in ["fp16", "no"]:
        raise ConfigError(f"system.mixed_precision must be 'fp16' or 'no', got {sys_cfg.get('mixed_precision')}")
