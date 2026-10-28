import csv
import json
import pathlib
import collections

ROOT = pathlib.Path(r"d:\Pneumonia_Detection\data\chest_xray")
MANIFESTS_DIR = pathlib.Path(r"d:\Pneumonia_Detection\data\manifests")

# 1. Read original dataset counts
original_counts = collections.defaultdict(lambda: collections.defaultdict(int))
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        p = ROOT / split / cls
        if p.exists():
            original_counts[split][cls] = len([f for f in p.iterdir() if f.is_file()])

# 2. Read manifests
def read_manifest(filename):
    p = MANIFESTS_DIR / filename
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

train_manifest = read_manifest("train_manifest.csv")
val_manifest = read_manifest("val_manifest.csv")
cal_manifest = read_manifest("calibration_manifest.csv")
test_manifest = read_manifest("test_manifest.csv")
val_ref_manifest = read_manifest("val_reference_16.csv")
dup_manifest = read_manifest("duplicates_manifest.csv")

all_manifest_records = train_manifest + val_manifest + cal_manifest + test_manifest + val_ref_manifest + dup_manifest

# Group by split and cls from manifest
manifest_counts = collections.defaultdict(lambda: collections.defaultdict(int))
for m, name in [(train_manifest, "train"), (val_manifest, "val"), (cal_manifest, "calibration"), 
                (test_manifest, "test"), (val_ref_manifest, "val_ref_16")]:
    for r in m:
        manifest_counts[name][r["label"]] += 1

dup_counts = collections.defaultdict(lambda: collections.defaultdict(int))
for r in dup_manifest:
    dup_counts[r["official_split"]][r["label"]] += 1

print("--- Original Counts ---")
total_original = 0
for split, cls_counts in original_counts.items():
    print(f"  {split}: NORMAL={cls_counts['NORMAL']}, PNEUMONIA={cls_counts['PNEUMONIA']}, Total={sum(cls_counts.values())}")
    total_original += sum(cls_counts.values())
print(f"  Total Original: {total_original}")

print("\n--- Manifest Counts ---")
total_manifest = 0
for split, cls_counts in manifest_counts.items():
    print(f"  {split}: NORMAL={cls_counts['NORMAL']}, PNEUMONIA={cls_counts['PNEUMONIA']}, Total={sum(cls_counts.values())}")
    total_manifest += sum(cls_counts.values())

print("\n--- Duplicates Manifest Counts (by official split) ---")
total_dups = 0
for split, cls_counts in dup_counts.items():
    print(f"  {split}: NORMAL={cls_counts['NORMAL']}, PNEUMONIA={cls_counts['PNEUMONIA']}, Total={sum(cls_counts.values())}")
    total_dups += sum(cls_counts.values())

print(f"\nTotal Manifest included: {total_manifest}")
print(f"Total Duplicates: {total_dups}")
print(f"Grand Total Manifests: {total_manifest + total_dups}")

# Verification of duplicate groups
dup_groups = collections.defaultdict(list)
all_original_images = []
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        p = ROOT / split / cls
        if p.exists():
            for f in p.iterdir():
                if f.is_file():
                    all_original_images.append((split, cls, f.name, f.relative_to(ROOT).as_posix()))

print(f"\nAll Original Images found recursively: {len(all_original_images)}")

# Check for unassigned or doubly assigned
assigned = collections.defaultdict(int)
for r in all_manifest_records:
    assigned[r["rel_path"]] += 1

unassigned = [p for _, _, _, p in all_original_images if assigned[p] == 0]
double_assigned = [p for p, c in assigned.items() if c > 1]

print(f"Unassigned images: {len(unassigned)}")
if unassigned:
    print(f"Sample unassigned: {unassigned[:5]}")
print(f"Double-assigned images: {len(double_assigned)}")
if double_assigned:
    print(f"Sample double_assigned: {double_assigned[:5]}")

# Duplicate breakdown
from collections import Counter
dup_groups_test = [r for r in dup_manifest if r["official_split"] == "test"]
dup_groups_train = [r for r in dup_manifest if r["official_split"] == "train"]
print(f"\nDuplicates in train: {len(dup_groups_train)}")
print(f"Duplicates in test: {len(dup_groups_test)}")

print("\n--- Patient Leakage Verification ---")
train_pat = set(r["patient_token"] for r in train_manifest if r["patient_token"])
val_pat = set(r["patient_token"] for r in val_manifest if r["patient_token"])
cal_pat = set(r["patient_token"] for r in cal_manifest if r["patient_token"])
test_pat = set(r["patient_token"] for r in test_manifest if r["patient_token"])

print(f"Train/Val overlap: {len(train_pat & val_pat)}")
print(f"Train/Cal overlap: {len(train_pat & cal_pat)}")
print(f"Val/Cal overlap: {len(val_pat & cal_pat)}")
print(f"Train/Test overlap: {len(train_pat & test_pat)} (Documented limitation)")

