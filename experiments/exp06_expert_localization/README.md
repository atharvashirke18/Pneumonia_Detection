# Expert-mask localization evaluation

No expert-annotated masks are currently included in this project, so no
localization scores are reported. Do not use generated, pseudo-labeled, or
Grad-CAM-derived masks as expert ground truth.

When masks become available, create a CSV with the columns shown in
`annotations_manifest_template.csv`:

- `image_id`: exact `rel_path` from `data/manifests/val_manifest.csv`
- `annotation_type`: `lung` or `lesion`
- `mask_path`: PNG/TIFF mask path relative to the CSV's directory
- `annotator_id`: non-identifying annotator code

Masks must have the same dimensions as the corresponding source image. Zero
pixels are background and non-zero pixels mark the annotation. Use
independently prepared expert annotations; record annotation instructions and
whether masks were adjudicated separately in the research record. Keep source
images and masks under appropriate data-use and privacy controls.

Run from the project root:

```powershell
.\.venv\Scripts\python.exe experiments\exp06_expert_localization\evaluate_localization.py `
  --annotations experiments\exp06_expert_localization\annotations_manifest.csv
```

The evaluator only accepts images in the internal validation manifest. It
computes native-grid peak occupancy, activation mass inside the annotation,
top-20%-cell occupancy, and top-cell recall. Metrics are computed on the
native 7 × 7 Grad-CAM grid rather than the interpolated display map. Small
lesions may be lost at this resolution; localization results must report this
limitation and must not be interpreted as clinical localization performance.
The pointing hit is undefined for an all-zero map, and the evaluator excludes
undefined hits from the corresponding aggregate.
