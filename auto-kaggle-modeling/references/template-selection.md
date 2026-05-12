# Template selection

Stage 2 ships templates under `assets/templates/`. The agent picks one based on `comp_profile.task_type` and `compute_env.specs`, copies it into the new run dir, then customizes the **copy**, never the source.

## Decision table

| comp_profile.task_type | Default template | When to override |
|---|---|---|
| `tabular-binary` | `tabular-lgbm` | Use `tabular-cat` if cardinality of categorical features is very high (>1000 levels in any column) |
| `tabular-multiclass` | `tabular-lgbm` | Same as above |
| `tabular-regression` | `tabular-lgbm` | Use `tabular-cat` for the same reason, or `tabular-xgb` if recon suggests heavy XGBoost usage in top kernels |
| `tabular-ranking` | `tabular-lgbm` (rank objective) | LightGBM has native LambdaRank; CatBoost has YetiRank |
| `image-classification` | `vision-timm` | Use a custom template if the comp ships a starter notebook with a non-standard backbone |
| `image-segmentation` | `vision-timm-seg` | Skeleton built around `segmentation_models_pytorch`; agent implements mask loading + RLE encoding |
| `image-detection` | `vision-det` | Skeleton supports `ultralytics` / `mmdet` / `torchvision`; agent picks one in `framework` config |
| `nlp-classification` | `nlp-hf` | Default to `microsoft/deberta-v3-base` unless recon says otherwise |
| `nlp-token-classification` | `nlp-hf` | Same, with token-classification head |
| `nlp-regression` | `nlp-hf` | Use regression head; comps like Common Lit prompted this category |
| `nlp-generation` | `nlp-hf` | Use seq2seq head; needs custom inference loop |
| `time-series-forecast` | `tabular-lgbm` (with time-aware CV) | For long horizons, consider a custom template with N-BEATS or TFT |
| `time-series-event-detection` | custom | Stage 2 ships a stub; agent writes a sliding-window event-detection head |
| `audio-classification` | `vision-timm` (spectrogram input) | Or HF Audio Classification head |
| `multimodal` | custom | No single template fits — agent composes |
| `graph` | custom | Use PyG or DGL in a custom template |
| `recommendation` | `tabular-lgbm` (LambdaRank) or `nlp-hf` (two-tower) | Depends on comp framing |
| `other` | custom | Always |

## Compute-env overrides

| compute_env.env | Adjustments |
|---|---|
| `kaggle-notebook` | Reduce default fold count from 10 to 5 if any template uses 10. Cap `seeds: [42]` (single seed). Templates that fetch model weights at runtime must do so from a Kaggle Datasets mount, not the internet (because submission run has no internet). |
| `cpu-only` | Refuse `vision-timm` and `nlp-hf` templates — they need CUDA. Only `tabular-lgbm` / `tabular-cat` / `tabular-xgb` allowed. Set `lgbm.device='cpu'` and reduce `n_estimators` if CV would exceed budget. |
| `local-gpu` | No restrictions beyond budget. Pre-flight check that VRAM >= template's `min_vram_gb` in `template.yaml`. |
| `cloud-gpu` | Same as local-gpu, plus checkpoint to a persistent location so a paused/resumed cloud instance can resume. |

## Per-template `template.yaml` schema

Every template directory has a `template.yaml` declaring its requirements:

```yaml
name: tabular-lgbm
applies_to: [tabular-binary, tabular-multiclass, tabular-regression, tabular-ranking]
min_vram_gb: 0            # CPU-friendly
base_hours_per_fold:
  tabular-binary: 0.2     # rough wallclock at default config on a 1M-row dataset
  tabular-multiclass: 0.3
  tabular-regression: 0.2
  tabular-ranking: 0.4
defaults:
  model: lgbm
  n_estimators: 5000
  early_stopping_rounds: 200
  num_leaves: 63
  learning_rate: 0.02
  feature_fraction: 0.9
  bagging_fraction: 0.85
  bagging_freq: 5
deps_python:
  - lightgbm>=4.0
  - pandas
  - numpy
  - pyyaml
  - scikit-learn
entry_point: train.py
internet_required_at_train: false
internet_required_at_infer: false
```

The agent reads this when picking a template and uses `base_hours_per_fold` in the budget estimator.

## When no template fits

For tasks tagged `other` or for niche subtypes (e.g. molecular property prediction, recommender systems with implicit feedback), the agent writes a one-off `train.py` directly in the run dir with `template: custom` in `config.yaml`. The custom train.py still must:

1. Read `cv_split.yaml` from `runs/<comp_slug>/stage2_modeling/cv_split.yaml`.
2. Write `progress.jsonl` events `fold_started` / `fold_done` for resume safety.
3. Save fold-N OOFs atomically before fold-N+1.
4. Produce `oof.npy`, `test_preds.csv`, `cv_score.json`, `train.log` at the end.

Templates exist to make this less typing, not because the contract changes — the contract is the same regardless.

## Template customization vs new template

When does an idea require a new template vs. a customization of an existing one?

- **Customization** (no new template): hyperparameter changes, new features, new augmentation policy, different fold count, different seed, target transform, ensemble of existing OOFs.
- **New template needed**: switching model family (LGBM → CatBoost → XGBoost; ResNet → ViT → ConvNeXt), switching framework (sklearn → torch), switching task head (classification → ranking).

A run with `template: tabular-lgbm` and `model: catboost` in `config.yaml` is a bug — that's a template mismatch. Pick the right template first.

## Templates currently shipped

Right now `auto-kaggle-modeling/assets/templates/` contains:

Fully functional:
- **`tabular-lgbm/`** — LightGBM with KFold / Stratified / Group split dispatch, target transforms, categorical handling, fold-by-fold atomic checkpoints.
- **`ensemble/`** — blends (arithmetic / geometric / rank-mean) or stacks (Ridge / LGBM) prior runs' OOFs + test preds. Produces a run with the same contract as model runs; cheap (seconds for tabular, minutes for vision).

Skeletons (outer loop runs; agent customizes model / dataset / metric / submission encoding in the run-dir copy):
- **`vision-timm/`** — image classification with timm backbones + albumentations.
- **`vision-timm-seg/`** — segmentation with `segmentation_models_pytorch` + SMP losses (Dice / BCE / Focal / Lovasz).
- **`vision-det/`** — detection with framework hook for `ultralytics` / `mmdet` / `torchvision`.
- **`nlp-hf/`** — HuggingFace transformer fine-tuning with mixed precision + linear warmup.

More templates land in later commits (planned: `tabular-cat`, `tabular-xgb`, `tabular-stack`).

## What the orchestrator records when picking a template

In the run's `config.yaml`:

```yaml
template: tabular-lgbm
template_source_path: auto-kaggle-modeling/assets/templates/tabular-lgbm
template_version_hash: <sha-1 of template dir at copy time>
```

The hash protects against silent template drift across runs (you'd expect identical-template runs to have identical infrastructure). Stage 3 uses this when assessing whether two runs are truly independent for blending.
