"""
build_manifests.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
Builds CSV manifests for train / val / calibration / test splits.

Strategy:
  - Test set   : official test/ — NEVER touched, loaded as-is.
  - Val-16     : official val/ — kept for reference only (16 images).
  - Patient grouping: PNEUMONIA images carry person<N> tokens.
    NORMAL images have IM-XXXX format with no patient token.
    Strategy: group PNEUMONIA by patient token; treat each NORMAL as its own unit.
  - From training data (5,216 images):
      * Within-train exact duplicates: keep first occurrence, flag rest
        in a separate manifest (do NOT delete original files).
      * De-duplicated train pool: 5,216 - 32 = 5,184 images.
      * Stratified split with patient-group constraints for PNEUMONIA:
          - 20% val+cal reserve (stratified, group-aware for PNEUMONIA)
          - Of the 20% reserve: 50% validation, 50% calibration
          - 80% goes to model training
  - Random seed: 42 (fixed, documented).
  - Leakage checks: verify no patient token appears in both train and val/cal.

Outputs (all in data/manifests/):
  - train_manifest.csv
  - val_manifest.csv
  - calibration_manifest.csv
  - test_manifest.csv
  - val_reference_16.csv
  - duplicates_manifest.csv
  - split_summary.json
  - leakage_report.json
"""

import json, re, hashlib, pathlib, random
import collections, csv
from datetime import datetime

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

ROOT       = pathlib.Path(r"d:\Pneumonia_Detection\data\chest_xray")
OUT_DIR    = pathlib.Path(r"d:\Pneumonia_Detection\data\manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
SPLITS_OFFICIAL = ["train", "val", "test"]
CLASSES        = ["NORMAL", "PNEUMONIA"]

# ── Step 1: Collect all images ────────────────────────────────────────────────
print("[1] Collecting all images...")

def extract_patient_token(filename):
    stem = pathlib.Path(filename).stem
    m = re.match(r'^(person\d+)', stem, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None   # NORMAL images have no reliable patient token

def md5_file(path, chunk=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf: break
            h.update(buf)
    return h.hexdigest()

all_records = []  # list of dicts
for official_split in SPLITS_OFFICIAL:
    for cls in CLASSES:
        d = ROOT / official_split / cls
        if not d.exists(): continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS: continue
            tok = extract_patient_token(f.name)
            all_records.append({
                "official_split": official_split,
                "label":          cls,
                "label_int":      0 if cls == "NORMAL" else 1,
                "patient_token":  tok,
                "filename":       f.name,
                "rel_path":       f.relative_to(ROOT).as_posix(),
                "abs_path":       str(f),
                "md5":            None,   # filled below
                "is_duplicate":   False,
                "duplicate_of":   None,
                "assigned_split": None,
            })

print(f"   Total records: {len(all_records)}")

# ── Step 2: MD5 hashing (for duplicate detection) ─────────────────────────────
print("[2] MD5 hashing all images (for duplicate manifest)...")
hash_map = collections.defaultdict(list)
for rec in all_records:
    h = md5_file(rec["abs_path"])
    rec["md5"] = h
    hash_map[h].append(rec)

# ── Step 3: Mark duplicates ───────────────────────────────────────────────────
print("[3] Marking duplicates...")
dup_groups = {h: recs for h, recs in hash_map.items() if len(recs) > 1}

duplicate_records = []
kept_md5s = set()

for h, recs in dup_groups.items():
    # Sort deterministically; keep the first (alphabetically by rel_path)
    recs_sorted = sorted(recs, key=lambda r: r["rel_path"])
    primary = recs_sorted[0]
    kept_md5s.add(h + "_" + primary["rel_path"])
    for rec in recs_sorted[1:]:
        rec["is_duplicate"] = True
        rec["duplicate_of"] = primary["rel_path"]
        duplicate_records.append(rec)

print(f"   Duplicate groups: {len(dup_groups)}  |  Redundant images: {len(duplicate_records)}")

# Active training pool = train images that are NOT marked as duplicates
train_pool = [r for r in all_records
              if r["official_split"] == "train" and not r["is_duplicate"]]
print(f"   Training pool after dedup: {len(train_pool)}")

# ── Step 4: Patient-grouped split of training pool ────────────────────────────
print("[4] Splitting training pool (patient-grouped for PNEUMONIA)...")

# Separate PNEUMONIA (has patient tokens) and NORMAL (no reliable token)
pneum_records = [r for r in train_pool if r["label"] == "PNEUMONIA"]
norm_records  = [r for r in train_pool if r["label"] == "NORMAL"]

# Group PNEUMONIA by patient token
pneum_groups = collections.defaultdict(list)
for r in pneum_records:
    pneum_groups[r["patient_token"]].append(r)
pneum_group_keys = sorted(pneum_groups.keys())  # deterministic

# NORMAL: each image is its own "group"
norm_group_keys = [r["rel_path"] for r in norm_records]

# We want ~20% of images for val+cal reserve.
# For PNEUMONIA: shuffle patient groups, split 80/20 by patient count.
# For NORMAL:    shuffle image list, split 80/20 by image count.

rng = random.Random(RANDOM_SEED)

# -- PNEUMONIA patient split
rng.shuffle(pneum_group_keys)
n_pneum_groups     = len(pneum_group_keys)
n_pneum_reserve_g  = max(1, round(n_pneum_groups * 0.20))
pneum_reserve_keys = set(pneum_group_keys[:n_pneum_reserve_g])
pneum_train_keys   = set(pneum_group_keys[n_pneum_reserve_g:])

pneum_reserve = [r for k in pneum_reserve_keys for r in pneum_groups[k]]
pneum_train   = [r for k in pneum_train_keys   for r in pneum_groups[k]]

# -- NORMAL image split
rng.shuffle(norm_records)
n_norm_reserve = max(1, round(len(norm_records) * 0.20))
norm_reserve   = norm_records[:n_norm_reserve]
norm_train     = norm_records[n_norm_reserve:]

# -- Combine
reserve_pool = pneum_reserve + norm_reserve
train_final  = pneum_train   + norm_train

print(f"   PNEUMONIA patient groups: {n_pneum_groups}  →  train:{len(pneum_train_keys)}  reserve:{len(pneum_reserve_keys)}")
print(f"   Train final  : {len(train_final)}  "
      f"(NORMAL={sum(1 for r in train_final if r['label']=='NORMAL')}  "
      f"PNEUMONIA={sum(1 for r in train_final if r['label']=='PNEUMONIA')})")
print(f"   Reserve pool : {len(reserve_pool)}  "
      f"(NORMAL={sum(1 for r in reserve_pool if r['label']=='NORMAL')}  "
      f"PNEUMONIA={sum(1 for r in reserve_pool if r['label']=='PNEUMONIA')})")

# ── Step 5: Split reserve 50/50 → val / calibration ──────────────────────────
# CRITICAL: Split by patient GROUP for PNEUMONIA (not by image) so no patient
# appears in both val and calibration.
rng2 = random.Random(RANDOM_SEED + 1)

# -- PNEUMONIA: split reserve patient groups 50/50
pneum_reserve_group_keys = list(pneum_reserve_keys)
rng2.shuffle(pneum_reserve_group_keys)
mid = len(pneum_reserve_group_keys) // 2
pneum_val_keys = set(pneum_reserve_group_keys[:mid])
pneum_cal_keys = set(pneum_reserve_group_keys[mid:])

pneum_val_list = [r for k in pneum_val_keys for r in pneum_groups[k]]
pneum_cal_list = [r for k in pneum_cal_keys for r in pneum_groups[k]]

# -- NORMAL: split reserve images 50/50 (no patient grouping available)
norm_res = [r for r in reserve_pool if r["label"] == "NORMAL"]
rng2.shuffle(norm_res)
norm_val_list = norm_res[:len(norm_res)//2]
norm_cal_list = norm_res[len(norm_res)//2:]

val_final = norm_val_list + pneum_val_list
cal_final = norm_cal_list + pneum_cal_list

print(f"   Val split    : {len(val_final)}  "
      f"(NORMAL={len(norm_val_list)}  PNEUMONIA={len(pneum_val_list)})")
print(f"   Cal split    : {len(cal_final)}  "
      f"(NORMAL={len(norm_cal_list)}  PNEUMONIA={len(pneum_cal_list)})")

# Assign split labels
for r in train_final: r["assigned_split"] = "train"
for r in val_final:   r["assigned_split"] = "val"
for r in cal_final:   r["assigned_split"] = "calibration"

# Official test
test_records = [r for r in all_records if r["official_split"] == "test"]
for r in test_records: r["assigned_split"] = "test"

# Official val-16
val16_records = [r for r in all_records if r["official_split"] == "val"]
for r in val16_records: r["assigned_split"] = "val_reference_16"

# ── Step 6: Leakage checks ────────────────────────────────────────────────────
print("[5] Running leakage checks...")

leakage_issues = []

# Check 1: No patient token in both train and val/cal (PNEUMONIA only)
train_tokens = set(r["patient_token"] for r in train_final
                   if r["patient_token"] is not None)
val_tokens   = set(r["patient_token"] for r in val_final
                   if r["patient_token"] is not None)
cal_tokens   = set(r["patient_token"] for r in cal_final
                   if r["patient_token"] is not None)
test_tokens  = set(r["patient_token"] for r in test_records
                   if r["patient_token"] is not None)

tv_overlap  = train_tokens & val_tokens
tc_overlap  = train_tokens & cal_tokens
tt_overlap  = train_tokens & test_tokens
vc_overlap  = val_tokens   & cal_tokens

print(f"   Train∩Val tokens (PNEUMONIA)  : {len(tv_overlap)}")
print(f"   Train∩Cal tokens (PNEUMONIA)  : {len(tc_overlap)}")
print(f"   Train∩Test tokens (PNEUMONIA) : {len(tt_overlap)}  ← official dataset overlap (expected)")
print(f"   Val∩Cal tokens (PNEUMONIA)    : {len(vc_overlap)}")

if tv_overlap:
    leakage_issues.append({"type": "PATIENT_TOKEN_TRAIN_VAL", "tokens": sorted(tv_overlap)})
if tc_overlap:
    leakage_issues.append({"type": "PATIENT_TOKEN_TRAIN_CAL", "tokens": sorted(tc_overlap)})
if vc_overlap:
    leakage_issues.append({"type": "PATIENT_TOKEN_VAL_CAL", "tokens": sorted(vc_overlap)})

# Check 2: No exact MD5 across train and val/cal
train_md5s  = set(r["md5"] for r in train_final)
val_md5s    = set(r["md5"] for r in val_final)
cal_md5s    = set(r["md5"] for r in cal_final)
test_md5s   = set(r["md5"] for r in test_records)

md5_tv  = train_md5s & val_md5s
md5_tc  = train_md5s & cal_md5s
md5_tt  = train_md5s & test_md5s
md5_vc  = val_md5s   & cal_md5s

print(f"   MD5 exact dups: train∩val={len(md5_tv)}  train∩cal={len(md5_tc)}  "
      f"train∩test={len(md5_tt)}  val∩cal={len(md5_vc)}")

for tag, s in [("MD5_TRAIN_VAL", md5_tv), ("MD5_TRAIN_CAL", md5_tc),
               ("MD5_VAL_CAL", md5_vc)]:
    if s:
        leakage_issues.append({"type": tag, "count": len(s)})

# Check 3: Total count
# Duplicates that belong to the test set are already counted in test_records,
# so exclude them from the duplicate addend to avoid double-counting.
train_dup_records = [r for r in duplicate_records if r["official_split"] == "train"]
expected_total = (len(train_final) + len(val_final) + len(cal_final) +
                  len(test_records) + len(val16_records) + len(train_dup_records))
count_ok = len(all_records) == expected_total
print(f"   Record count check: all={len(all_records)}  accounted={expected_total}  OK={count_ok}")
if not count_ok:
    leakage_issues.append({"type": "COUNT_MISMATCH",
                            "all": len(all_records), "accounted": expected_total,
                            "difference": expected_total - len(all_records)})

# ── Step 7: Write CSV manifests ───────────────────────────────────────────────
print("[6] Writing CSV manifests...")

FIELDNAMES = ["assigned_split", "official_split", "label", "label_int",
              "patient_token", "filename", "rel_path", "abs_path",
              "md5", "is_duplicate", "duplicate_of"]

def write_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in FIELDNAMES})
    print(f"   Wrote {len(records):5d} rows → {path.name}")

write_csv(OUT_DIR / "train_manifest.csv",       train_final)
write_csv(OUT_DIR / "val_manifest.csv",          val_final)
write_csv(OUT_DIR / "calibration_manifest.csv",  cal_final)
write_csv(OUT_DIR / "test_manifest.csv",          test_records)
write_csv(OUT_DIR / "val_reference_16.csv",       val16_records)
write_csv(OUT_DIR / "duplicates_manifest.csv",    duplicate_records)

# ── Step 8: Summary JSON ──────────────────────────────────────────────────────
def cls_counts(recs):
    return {"NORMAL": sum(1 for r in recs if r["label"]=="NORMAL"),
            "PNEUMONIA": sum(1 for r in recs if r["label"]=="PNEUMONIA"),
            "total": len(recs)}

summary = {
    "generated_at":   datetime.now().isoformat(),
    "random_seed":    RANDOM_SEED,
    "dataset_root":   str(ROOT),
    "strategy": {
        "test":         "Official test/ — unchanged, reserved for final evaluation only.",
        "val_ref_16":   "Official val/ — 16 images, kept for reference only.",
        "train_reserve_pct": 0.20,
        "reserve_split": "50% val / 50% calibration (stratified by class within reserve).",
        "pneumonia_grouping": "Patient-level grouping by person<N> token. No patient token appears in both train and val/calibration.",
        "normal_grouping":   "Image-level split (no reliable patient token in NORMAL filenames).",
        "duplicate_policy":  "32 redundant images flagged in duplicates_manifest.csv. Original files untouched. Only first occurrence per MD5 group is in train_manifest.",
    },
    "counts": {
        "train":           cls_counts(train_final),
        "val":             cls_counts(val_final),
        "calibration":     cls_counts(cal_final),
        "test":            cls_counts(test_records),
        "val_reference_16": cls_counts(val16_records),
        "duplicates_excluded": len(duplicate_records),
    },
    "leakage_check": {
        "patient_token_train_val_overlap":  len(tv_overlap),
        "patient_token_train_cal_overlap":  len(tc_overlap),
        "patient_token_train_test_overlap": len(tt_overlap),
        "patient_token_val_cal_overlap":    len(vc_overlap),
        "md5_exact_train_val":              len(md5_tv),
        "md5_exact_train_cal":              len(md5_tc),
        "md5_exact_val_cal":                len(md5_vc),
        "issues_found":                     len(leakage_issues),
        "issues":                           leakage_issues,
    },
    "note_train_test_token_overlap": (
        f"{len(tt_overlap)} PNEUMONIA patient tokens appear in both the new train split and "
        "the official test set. This is the known Kaggle dataset characteristic: different "
        "images (confirmed by MD5), same patient. Images are not duplicated. This is documented "
        "as a limitation; the official test set is kept unchanged per project spec."
    ),
}

summary_path = OUT_DIR / "split_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"   Wrote summary → {summary_path.name}")

leakage_path = OUT_DIR / "leakage_report.json"
with open(leakage_path, "w", encoding="utf-8") as f:
    json.dump(summary["leakage_check"], f, indent=2)
print(f"   Wrote leakage report → {leakage_path.name}")

# ── Final report ──────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("MANIFEST GENERATION COMPLETE")
print("=" * 65)
print(f"  Train       : {cls_counts(train_final)}")
print(f"  Val         : {cls_counts(val_final)}")
print(f"  Calibration : {cls_counts(cal_final)}")
print(f"  Test        : {cls_counts(test_records)}  [official — unchanged]")
print(f"  Val-ref-16  : {cls_counts(val16_records)}  [official — reference only]")
print(f"  Duplicates excluded: {len(duplicate_records)} (files untouched)")
print()
print(f"  Leakage issues found: {len(leakage_issues)}")
if leakage_issues:
    for iss in leakage_issues:
        print(f"    *** {iss}")
else:
    print("  [PASS] No patient-token or MD5 leakage between train/val/cal.")
print()
print(f"  Random seed : {RANDOM_SEED}")
print(f"  Output dir  : {OUT_DIR}")
print("=" * 65)
