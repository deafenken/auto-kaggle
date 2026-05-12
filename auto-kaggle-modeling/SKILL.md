---
name: auto-kaggle-modeling
description: >-
  Stage 2 of auto-kaggle (NOT YET IMPLEMENTED — design only). Builds the
  user's own CV-aware pipeline for the competition, picking model templates
  from compute_env.yaml (tabular LightGBM/XGBoost/CatBoost; CV timm/HF;
  NLP HF transformers). Integrates ideas from stage1_recon/ideas_pool.md
  one at a time with ablations so we know which ideas actually helped.
  Tracks every run's CV vs public-LB gap in leaderboard.csv. Refuses to
  start runs that exceed compute_env.constraints.max_wallclock_per_run_hours.
---

# Stage 2 — Modeling (DESIGN PLACEHOLDER)

> **Status:** contract only — implementation lands in the next delivery.

## Trigger

- Delegated to by `auto-kaggle` after recon completes a cycle, or whenever new ideas land in `ideas_pool.md`.
- Can be invoked directly: `auto-kaggle-modeling <comp_slug>`.

## Inputs

- `runs/<comp_slug>/stage0_bootstrap/comp_profile.yaml`
- `runs/<comp_slug>/stage0_bootstrap/compute_env.yaml`
- `runs/<comp_slug>/stage1_recon/ideas_pool.md`
- `runs/<comp_slug>/stage2_modeling/leaderboard.csv` (if exists)
- `runs/<comp_slug>/stage2_modeling/cv_split.yaml` (if exists; if not, this stage creates it on first run)

## Outputs (contract)

```
runs/<comp_slug>/stage2_modeling/
├── pipeline.py                  # user-owned pipeline, re-implemented from ideas
├── cv_split.yaml                # CV scheme + rationale (group/time/stratified) — set once, user-changes only
├── runs/<run_id>/
│   ├── config.yaml              # all hyperparameters, seeds, fold count, features used
│   ├── oof.npy                  # out-of-fold predictions on train
│   ├── test_preds.csv           # predictions on test in submission format
│   ├── cv_score.json            # per-fold + aggregate CV + variance
│   ├── attribution.md           # which recon ideas this run uses + own additions
│   └── train.log
├── leaderboard.csv              # rolling: run_id, cv_score, public_lb, gap, status
└── hand_off.md                  # briefing for Stage 3
```

`<run_id>` is `YYYY-MM-DD-<short-slug>`, e.g. `2026-05-12-lgbm-baseline`.

## Workflow (planned)

1. On first invocation: write `cv_split.yaml` based on `comp_profile.task_type` (StratifiedKFold for classification, KFold for regression, GroupKFold if group columns detected, TimeSeriesSplit if temporal). Record rationale.
2. On every invocation: read the latest `ideas_pool.md`; identify ideas not yet tried (cross-checked against past `attribution.md` files).
3. Pick the next experiment per a priority rule (TODO: define) — usually: largest expected CV gain × smallest implementation effort.
4. Estimate wallclock; refuse if it exceeds `compute_env.constraints.max_wallclock_per_run_hours`. Escalate per integrity rule 8.
5. Create `runs/<run_id>/`. Train. Log to `progress.jsonl` event `fold_done` after each fold (so resume can pick up mid-run).
6. Compute CV score. Append a row to `leaderboard.csv`.
7. Write `attribution.md` naming source kernels and own deltas.
8. Update `hand_off.md` briefing Stage 3 with: best CV score, this run's gap vs prior best, whether a submission is recommended now.

## Integrity gates this stage enforces

- Rule 1 — every borrowed technique is re-implemented in `pipeline.py` with a `# Derived from: <author>/<kernel-slug>` comment.
- Rule 7 — `cv_split.yaml` is never modified by this stage after the first write. If the agent thinks the split is wrong, it escalates per `escalation-policy.md` instead of editing.
- Rule 8 — wallclock estimator must run before starting any new run.

## Templates (planned `assets/templates/`)

- `tabular-lgbm/` — LightGBM with KFold, Optuna-ready
- `tabular-cat/` — CatBoost
- `tabular-xgb/` — XGBoost
- `tabular-stack/` — stacking of the above
- `vision-timm/` — timm pretrained + albumentations + TTA
- `vision-mmseg/` — segmentation
- `nlp-hf/` — HuggingFace transformers + trainer + KFold
- `timeseries-lgbm/` — leak-aware time series with TimeSeriesSplit

Choice is driven by `comp_profile.task_type` and `compute_env.specs`.

## When to load which reference (planned)

| File | Load when |
|---|---|
| `auto-kaggle/references/state-contract.md` | Always |
| `auto-kaggle/references/integrity-rules.md` | Before starting a new run and at every fold boundary |
| `auto-kaggle/references/long-running-protocol.md` | Resume / heartbeat / fold-by-fold idempotency |
| `auto-kaggle/references/external-tools.md` | When delegating feature brainstorm or hyperparameter search |
| `auto-kaggle/references/escalation-policy.md` | Considering changing `cv_split.yaml` or exceeding budget |
