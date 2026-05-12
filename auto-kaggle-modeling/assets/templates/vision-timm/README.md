# vision-timm template

Skeleton for vision competitions using `timm` pretrained backbones + `albumentations` augmentations.

**Status:** skeleton. The functions `build_model`, `build_datasets`, `build_test_dataset`, and `compute_metric` raise `NotImplementedError` and must be filled in for the specific competition. The outer training loop (optimizer / scheduler / mixed precision / fold-by-fold checkpoint / progress logging) is functional.

The agent edits the COPY of this template in `runs/<comp_slug>/stage2_modeling/runs/<run_id>/`, never the source.

## Why a skeleton instead of a full template

Vision competitions vary far more than tabular: image loading paths, mask formats for segmentation, multi-label vs single-label heads, custom metrics (IoU vs Dice vs mAP). A single concrete template would either be wrong for half the competitions or so configurable it's unreadable. The skeleton commits to:

- Outer loop shape (fold/seed iteration, optimizer/scheduler, AMP)
- Atomic per-fold checkpoints + resume
- Progress.jsonl events
- `cv_score.json` schema at the end

…and leaves the actual model + dataloader to the agent's per-comp edit.

## What the agent must implement

```python
def build_model(cfg) -> torch.nn.Module:
    import timm
    return timm.create_model(
        cfg["model_name"], pretrained=cfg["pretrained"],
        num_classes=<comp specific>,
    )

def build_datasets(cfg, train_df, (tr_idx, va_idx)) -> tuple[Dataset, Dataset]:
    # Build torch Datasets from train_df rows. Each item must yield a dict with
    # at least 'image' (FloatTensor [C, H, W]) and 'target' (LongTensor or FloatTensor).
    ...

def build_test_dataset(cfg, test_df) -> Dataset:
    ...

def compute_metric(metric_name, y_true, y_pred) -> float:
    # Match comp_profile.metric.name and apply that metric.
    ...
```

The agent can also adjust `_train_one_epoch` and `_infer` if the competition needs:
- Mixup / CutMix (insert before forward pass)
- TTA at inference (run inference N times with different aug, average)
- Segmentation-style outputs (replace softmax with whatever the head emits)

## Config (`template.yaml`)

Defaults: ConvNeXt-base from timm, 384×384, 10 epochs, AdamW + cosine LR, AMP on.

Override via `config.yaml` overrides:
- `model_name` — any timm name; commonly `eva02_large_patch14_clip_336`, `convnext_large.fb_in22k_ft_in1k`, `swinv2_base_window12to24_192to384`, etc.
- `image_size` — input resolution; affects VRAM quadratically
- `epochs` / `batch_size` / `lr` — usual training knobs
- `augmentation.policy` — `none` / `light` / `medium` / `heavy` (agent translates to a concrete albumentations Compose)
- `tta.enabled` / `tta.n_views` — test-time augmentation

## Compute requirements

- Default config needs ~8 GB VRAM. Larger backbones (eva02-L, swinv2-L) need 24 GB.
- 80k train images × 384² × 10 epochs ≈ 2 hours per fold on a single 3090. Scale linearly with image count.

## Mandatory pre-training checks

Before invoking, the agent verifies:
- The competition's pretrained-weights rule (Rule 10): some comps require pretrained-weights-cutoff dates. `timm.create_model(pretrained=True)` downloads recent weights — if the cutoff is older than the pretrained release date, the run is non-compliant.
- VRAM: `torch.cuda.get_device_properties(0).total_memory >= template.min_vram_gb * 1e9`.
- `data/raw/` has the expected image folder structure.

## What this template does NOT cover

- Segmentation (mask handling, RLE encoding) — write a fork `vision-timm-seg/` with the appropriate head and loss.
- Object detection — write `vision-mmdet/` or `vision-ultralytics/`.
- Multi-stage / coarse-to-fine training pipelines — custom train.py in the run dir.
