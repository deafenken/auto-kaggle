# vision-det template

Skeleton for object detection competitions. Supports three frameworks via a hook pattern; agent picks one and customizes it for the comp.

**Status:** skeleton. Outer loop (fold iteration, progress events, cv_score.json) is functional. The framework-specific functions `train_one_fold_*`, `convert_annotations_for_framework`, and `encode_submission` raise `NotImplementedError`.

## Framework choice

| Framework | When to use | Pros | Cons |
|---|---|---|---|
| `ultralytics` | First baseline, simple comps | Fastest setup, single-line train, decent results | Limited to YOLO family, less control |
| `mmdet` | Comps where you want DETR / DINO / Cascade-RCNN | Huge model zoo, configurable | Heavy install, config-file complexity |
| `torchvision` | Comps that ban large frameworks | Minimal deps, just torch | Smaller model zoo, more boilerplate |

Most agents start with `ultralytics` (YOLO v8 / v10) and switch to `mmdet` only if a stronger model is needed.

## What the agent must implement

```python
def convert_annotations_for_framework(cfg, train_df, (tr_idx, va_idx)):
    # Translate comp's annotation format (CSV with bbox columns, or COCO JSON,
    # or per-image XML) into the framework's expected layout.
    # For ultralytics: write a data.yaml + per-image .txt labels.
    # For mmdet: write COCO-format JSON for train and val.
    # Return whatever `train_one_fold_<framework>` needs.
    ...

def train_one_fold_ultralytics(cfg, paths):
    from ultralytics import YOLO
    model = YOLO(cfg["model_name"])
    results = model.train(
        data=paths.data_yaml,
        epochs=cfg["epochs"],
        imgsz=cfg["image_size"],
        batch=cfg["batch_size"],
        # ... other knobs from cfg["loss"], cfg["augmentation"]
    )
    val_map = results.box.map  # COCO mAP
    test_pred = model.predict(paths.test_images_dir, save=False)
    return val_map, encode_predictions_to_array(test_pred)

def encode_submission(cfg, test_predictions):
    # Per-image: build a "PredictionString" like "0.9 100 50 50 80  0.8 ..."
    # (label, conf, x, y, w, h order varies — check sample_submission.csv).
    ...
```

## Key config knobs

| Key | Default | Notes |
|---|---|---|
| `framework` | `ultralytics` | switch by editing config.yaml |
| `model_name` | `yolov8x` | ultralytics: `yolov8n` (fastest, weakest) → `yolov8x` (strongest); mmdet: config filename |
| `image_size` | 1280 | detection benefits from large resolution; VRAM grows linearly |
| `epochs` | 30 | comp deadline scales this |
| `augmentation.mosaic` | 1.0 | YOLO's mosaic aug; turn off in last 10 epochs (handled by ultralytics) |
| `postprocess.iou_threshold` | 0.5 | NMS threshold |
| `postprocess.conf_threshold` | 0.001 | low for max recall; comps using mAP score want this low |
| `tta.enabled` | false | detection TTA is expensive; only worth it pre-deadline |

## CV considerations

- Detection comps usually use GroupKFold on **scene / video / patient** when applicable. Naive KFold leaks if the same scene appears in multiple folds.
- `n_folds: 5` is standard but expensive (5× the already-long detection training).
- For very heavy comps, consider 3 folds + multi-seed instead.

## Metric

Most detection comps use mAP (mean Average Precision). Variants:
- `mAP@0.5` — IoU threshold 0.5 only (PASCAL VOC style)
- `mAP@[.5:.95]` — average over IoU thresholds 0.5 to 0.95 step 0.05 (COCO style; harder)
- Per-class mAP averaged

`comp_profile.metric.description` should record which. Default skeleton assumes COCO-style.

## Compute requirements

- YOLOv8x at 1280×1280: ~14 GB VRAM. Batch 8.
- mmdet DINO / Co-DETR: ~24 GB VRAM. Batch 2–4.
- Per-fold wallclock: 5–8 hours on a single 3090 for a 30k-image dataset, 30 epochs.

This is the heaviest template. Be aggressive with the budget estimator.

## Pseudo-labeling

Detection benefits a lot from pseudo-labeling — train on train + high-confidence predictions on test. Standard pattern:

1. Train baseline on train.
2. Predict on test; keep boxes with `conf > 0.9`.
3. Re-train on (train + pseudo-labeled test).
4. Repeat 1–2 more iterations.

Each iteration is a separate Stage 2 run with `attribution.md` noting "pseudo-labeled from <prior run>". Idea key: `aug:pseudo-labeling`.

## What this template does NOT cover

- Instance segmentation (Mask R-CNN with mask heads). Possible via mmdet config, but expect to write more custom code.
- Multi-frame / video detection (tracking comps).
- 3D detection (autonomous driving comps).
- Polygon-style detection (rotated boxes, oriented bounding boxes).
