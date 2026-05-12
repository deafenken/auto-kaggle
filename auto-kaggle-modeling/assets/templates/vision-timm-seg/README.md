# vision-timm-seg template

Skeleton for image segmentation competitions using `segmentation_models_pytorch` (smp) + timm encoders.

**Status:** skeleton. `build_datasets`, `build_test_dataset`, `encode_submission` raise `NotImplementedError` and must be implemented per competition (mask formats, RLE encoding, multi-class vs binary differ a lot).

`build_model` has a working default that wraps any timm encoder in an smp architecture (Unet / FPN / DeepLabV3+ / UnetPlusPlus / etc). `compute_metric` covers IoU and Dice; add anything custom in the run-dir copy.

## What the agent must implement

```python
def build_datasets(cfg, train_df, (tr_idx, va_idx)):
    # Use albumentations with mask=True so geometric augs apply consistently.
    # Each item yields {"image": FloatTensor[3,H,W], "mask": FloatTensor[classes,H,W]}.
    # Also expose val_ds.true_masks (or similar) so the outer loop can compute CV.
    ...

def build_test_dataset(cfg, test_df):
    # Test items yield {"image": FloatTensor, "image_id": str}.
    ...

def encode_submission(cfg, test_ids, masks_logits):
    # Most segmentation comps use RLE. Standard pattern:
    #   1. apply threshold (default 0.5; tune on OOF after first fold)
    #   2. convert binary mask to RLE string
    #   3. produce DataFrame with columns matching sample_submission.csv
    ...
```

## Key config knobs

| Key | Default | Notes |
|---|---|---|
| `framework` | `smp` | Switch to `custom` to use your own architecture |
| `architecture` | `Unet` | SMP names: `Unet`, `UnetPlusPlus`, `FPN`, `DeepLabV3Plus`, `MAnet`, `PAN`, `PSPNet` |
| `encoder_name` | `tu-convnext_base.fb_in22k_ft_in1k` | Any timm via SMP's `tu-` prefix |
| `image_size` | 768 | Larger = better quality but VRAM × size² |
| `classes` | 1 | 1 for binary masks; ≥2 for multi-class |
| `loss.type` | `bce_dice` | `bce_dice` / `dice` / `focal` / `lovasz` |
| `submission.encoding` | `rle` | Most comps use RLE; PNG paths are rarer |
| `submission.threshold` | 0.5 | Auto-tune on OOF later for ~+0.5 to +2 IoU |

## CV considerations

Segmentation usually benefits from **GroupKFold on patient/series/scene**, not StratifiedKFold. Set `cv_split.scheme: GroupKFold` and identify the group column in `data_stats.md`. Multiple masks per scene → group on scene id.

If masks come from multiple annotators, consider whether to stratify on annotator (some comps have known label noise patterns by source).

## Threshold tuning

Most seg comps' metric (IoU at threshold τ) is sensitive to τ. After the first fold:

1. Load OOF probabilities for the validation fold.
2. Sweep τ ∈ {0.30, 0.35, ..., 0.70}.
3. Pick the τ that maximizes IoU on OOF.
4. Lock it into `submission.threshold` for all subsequent folds AND for test inference.

Tuning τ on OOF and then using it on test is fine — it does not leak. Tuning per-fold and changing per-test would.

## Compute requirements

- 768×768 with ConvNeXt-base + Unet: ~10 GB VRAM at batch 8. Cut batch or image_size if VRAM-constrained.
- 50k train images × 768² × 30 epochs ≈ 4 hours per fold on a single 3090.
- Mixed precision is essential for any non-tiny seg task.

## TTA

`tta.enabled: true` runs each test image through `[identity, hflip, vflip, hflip+vflip]` and averages predictions. Worth +0.5 to +1 IoU on most comps.

## Mandatory pre-training checks

- Rule 10: pretrained-weights cutoff. Many comps disallow ImageNet-22k pretraining; verify in `comp_profile.rules_summary`.
- VRAM: `torch.cuda.get_device_properties(0).total_memory >= template.min_vram_gb * 1e9`.
- Mask format: dump a few train masks and confirm dtype (uint8 vs float, 0–1 vs 0–255).

## What this template does NOT cover

- Instance segmentation (Mask R-CNN-style). Use `vision-mmdet/` (planned).
- Panoptic segmentation. Custom train.py.
- 3D / volumetric segmentation (medical comps). Custom train.py with monai or similar.
- Pseudo-labeling for test set. Run as a separate Stage 2 run after the baseline lands.
