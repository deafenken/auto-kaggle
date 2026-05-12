---
name: auto-kaggle-modeling
description: >-
  Stage 2 of auto-kaggle. Builds the user's own CV-aware training pipeline,
  picking a template based on comp_profile.task_type and compute_env (tabular
  LightGBM/CatBoost/XGBoost; vision timm + albumentations; NLP HF transformers).
  Reads ideas from stage1_recon/ideas_pool.md, prioritizes by consensus and
  cost, integrates each idea as a separate ablation so we know which ones
  pulled weight, and tracks every run's CV score in leaderboard.csv. Refuses
  to start runs that exceed compute_env wallclock budgets. Crash-safe via
  fold-by-fold progress.jsonl events so a killed run resumes from the next
  fold, not from scratch.
---

# Stage 2 — Modeling

The only stage that actually trains models. Reads ideas from recon, builds the user's own pipeline (no verbatim copy), runs experiments with one idea changed at a time so ablations are clean, and writes results into a leaderboard Stage 3 reads.

## Trigger

- Delegated to by `auto-kaggle` after recon completes a cycle, or whenever the orchestrator decides we have idle compute and untried ideas.
- Direct: `auto-kaggle-modeling <comp_slug>` to start one or more new runs.
- `auto-kaggle-modeling status <comp_slug>` prints `leaderboard.csv` and exits.

## Inputs

- `runs/<comp_slug>/stage0_bootstrap/comp_profile.yaml`
- `runs/<comp_slug>/stage0_bootstrap/compute_env.yaml`
- `runs/<comp_slug>/stage0_bootstrap/data_stats.md`
- `runs/<comp_slug>/stage1_recon/ideas_pool.md`
- `runs/<comp_slug>/stage1_recon/kernels_index.json`
- `runs/<comp_slug>/stage2_modeling/leaderboard.csv` (if exists)
- `runs/<comp_slug>/stage2_modeling/cv_split.yaml` (if exists)
- `runs/<comp_slug>/stage2_modeling/runs/*/attribution.md` (for ideas-already-tried diffing)

## Outputs (contract — full schemas in `auto-kaggle/references/state-contract.md`)

```
runs/<comp_slug>/stage2_modeling/
├── pipeline.py                  # user-owned pipeline (re-implementations of recon ideas)
├── cv_split.yaml                # CV scheme + rationale, set once, user-changes only
├── runs/<run_id>/
│   ├── config.yaml              # hyperparameters, seeds, fold count, features used
│   ├── train.py                 # the run's frozen training entry point (copied from template)
│   ├── oof.npy                  # out-of-fold predictions on train (n_train,) or (n_train, n_classes)
│   ├── test_preds.csv           # predictions on test in submission format
│   ├── cv_score.json            # per-fold + aggregate CV + variance
│   ├── attribution.md           # which recon ideas this run uses + own additions
│   └── train.log                # stdout/stderr from the training subprocess
├── leaderboard.csv              # rolling: run_id, ts_utc, cv_score, cv_std, public_lb, status
├── feature_ideas.md             # optional brainstormed feature list (external-tools handoff #3)
└── hand_off.md                  # briefing for Stage 3
```

`<run_id>` is `YYYY-MM-DD-<short-kebab-slug>`, e.g. `2026-05-12-lgbm-baseline`.

## Workflow

### Step 0 — first invocation: set CV

If `cv_split.yaml` does not exist, derive a default from `comp_profile.task_type`:

| Task type | CV scheme | Note |
|---|---|---|
| tabular-binary / multiclass | `StratifiedKFold(n=5, shuffle=True, seed=42)` | stratify on target |
| tabular-regression | `KFold(n=5, shuffle=True, seed=42)` | unless `data_stats.md` flags a group column |
| (any) with detected group column | `GroupKFold(n=5)` | group column inferred from `data_stats.md` |
| time-series-* | `TimeSeriesSplit(n=5)` or domain-specific blocked split | rationale required |
| image-segmentation / detection | `GroupKFold` by image_id or patient_id | check `data_stats.md` |
| nlp-* | `StratifiedKFold` on label distribution | unless text source forms groups |

Record the choice with rationale:

```yaml
scheme: StratifiedKFold
n_folds: 5
shuffle: true
seed: 42
group_col: null
stratify_col: target
rationale: >-
  Tabular binary classification with no detected group column. Stratify on target
  to ensure class balance is consistent across folds. Seed fixed to 42 for
  reproducibility. User must edit this file (not the agent) to change the scheme.
```

Once written, Rule 7 — never modify on agent's own initiative.

### Step 1 — pick the next experiment

Read `ideas_pool.md`. Build the candidate list:

```
tried_idea_keys = union(attribution.md::ideas_used across all stage2_modeling/runs/*)
untried = ideas_pool.entries - tried_idea_keys
```

Priority score per untried idea:

```
score = (consensus_count * 2 if entry.consensus else 0)
      + (1 if entry.cost == 'S' else 0.5 if entry.cost == 'M' else 0.2)
      + (1 if entry.effect_reported else 0)
      + (0.5 if category_underused_in_tried else 0)
```

Where `category_underused_in_tried` means we have < 2 runs in that category. The portfolio balancing prevents 5 ensemble experiments in a row while features go untried.

Pick the highest-scored untried idea. If multiple tie, pick the smallest-cost one. If `untried` is empty, switch to "improvement mode": run a new seed of the current best run, OR ensemble the top 3 runs by CV (whichever has higher expected value).

### Step 2 — pick a template + run_id

`assets/templates/` has 3 currently shipped:

- `tabular-lgbm/` — LightGBM + KFold; the default for tabular tasks
- `vision-timm/` — timm pretrained + albumentations; default for image tasks (skeleton)
- `nlp-hf/` — HuggingFace transformers; default for NLP tasks (skeleton)

Selection table in `references/template-selection.md`. The agent can also write a one-off `train.py` inline if no template fits — in that case `template: custom` in `config.yaml`.

`run_id` is `<UTC date>-<kebab summary>`, e.g. `2026-05-12-lgbm-baseline`, `2026-05-12-lgbm-with-log-target`.

### Step 3 — wallclock estimate + budget gate

Before any training:

```python
estimate_h = (
    template.base_hours[task_type]
    * (n_folds / 5)
    * (data_size_factor)         # bigger data → linear-ish
    * (seed_count)
    * (1.0 if cost == 'S' else 1.5 if cost == 'M' else 2.5)
)
budget_h = compute_env.constraints.max_wallclock_per_run_hours
if estimate_h > budget_h:
    escalate(integrity_rule=8, estimate=estimate_h, budget=budget_h)
```

Estimation method in `references/budget-estimator.md`. Always err high (pad 1.3×) so we're not pleasantly surprised but never blow the budget.

### Step 4 — create the run, copy template, write config

```bash
mkdir -p runs/<slug>/stage2_modeling/runs/<run_id>/
cp -r auto-kaggle-modeling/assets/templates/<template>/* \
      runs/<slug>/stage2_modeling/runs/<run_id>/
```

Write `config.yaml` with:

- `template: tabular-lgbm`
- `task_type`, `metric`, `cv_scheme` (copied from upstream files)
- `seeds: [42]` (or more if compute_env allows)
- `idea_keys: [feature:log1p_target_then_xgboost, cv:stratified-kfold-5, ...]`
- `model_params: {...}` (template-specific)
- `feature_config: {...}` (which features to compute / exclude)
- `data_paths: {train: runs/<slug>/data/raw/train.csv, ...}`

Write `attribution.md` naming every kernel cited by the chosen ideas (from `ideas_pool.md`). Empty attribution is a Rule 2 violation and blocks Stage 3 from submitting this run.

### Step 5 — train (subprocess, fold-by-fold checkpointed)

Run the template's `train.py` as a subprocess:

```bash
cd runs/<slug>/stage2_modeling/runs/<run_id>/
python train.py --config config.yaml \
                --runs-dir ../../../../ \
                --comp-slug <slug> > train.log 2>&1
```

The training script is required to:

1. Write `progress.jsonl` events `fold_started` and `fold_done` (with per-fold CV score) after each fold completes.
2. Save fold-N OOF predictions atomically before moving on, so a crash mid-fold-N+1 is recoverable.
3. Skip a fold if its OOF slice file already exists with a matching config hash (idempotency).
4. Emit `cv_score.json` only after **all** folds complete.

The orchestrator polls `progress.jsonl` for `fold_done` events (heartbeat is still updated by the parent agent every 60s).

### Step 6 — finalize the run

After the subprocess returns:

1. Verify all fold OOFs exist and concatenate to `oof.npy`.
2. Compute aggregate CV (per `comp_profile.metric.name`) and variance across folds.
3. Generate `test_preds.csv` in submission format (template-specific).
4. Write `cv_score.json`:
   ```json
   {
     "metric": "rmse",
     "per_fold": [0.7432, 0.7401, 0.7448, 0.7415, 0.7398],
     "mean": 0.7419,
     "std": 0.0019,
     "n_train": 145000,
     "n_test": 50000
   }
   ```
5. Append to `leaderboard.csv` (`assets/leaderboard.py` handles the append).
6. Update `attribution.md` with: actual CV, gap vs prior best, ideas that earned their keep.
7. Append `progress.jsonl` event `run_finished` with `{run_id, cv_score, cv_std}`.

### Step 7 — hand-off

Write `hand_off.md`:

```markdown
# Stage 2 → Stage 3 hand-off (run <run_id> finished at <ts>)

## What I did
- Ran <run_id> using template <template> with ideas {<idea_keys>}.
- CV <metric> = <score> (std <std>) across <n_folds> folds.
- Ranked <position> on the local leaderboard (best is <best_run_id> at <best_score>).

## What's true now
- Total runs: <N>. Best CV: <score> by <run_id>.
- Trustworthy (cv_std < threshold) candidates: <list>.
- Ideas untried in idea pool: <count>.
- Compute used this cycle: <hours>. Remaining today: <hours>.

## What you should do next
Stage 3: refresh recommendations.md ranking by trust-adjusted CV.
If <run_id> beats current best by more than 1× cv_std, it should appear
near the top. Quota check: <used>/<limit> used today.
```

Update `.heartbeat` and exit.

## "Improvement mode" (when ideas_pool is exhausted)

When `untried` is empty, the agent runs one of:

- **More seeds**: re-run the current best with a different seed (`seed: 1337`). Variance reduction.
- **Larger fold count**: re-run with 10 folds instead of 5, for a tighter CV estimate.
- **Blend**: take the top 3 OOFs by CV, fit a non-negative least-squares blend, write the resulting `test_preds.csv` as a new "ensemble" run.
- **Stack**: fit a small Ridge or LGBM on top of OOFs as features.

Each of these is its own run with its own attribution. Improvement mode never re-uses an idea from a non-best run.

## Integrity gates this stage enforces

- Rule 1 — every borrowed technique is re-implemented in `pipeline.py` (or in the template). The kernels under `stage1_recon/kernels/` are read-only reference.
- Rule 7 — `cv_split.yaml` is written once; agent never re-edits, only user does. If the agent thinks the split is wrong, escalate per `escalation-policy.md`.
- Rule 8 — wallclock estimator runs before every new run; over-budget → escalate.
- Rule 10 — external data may only be loaded if its entry in `external_data_candidates.md` has `Approved for use: YES`.

## Templates (current and planned)

Shipped:
- `tabular-lgbm/` — fully functional LightGBM training entry point.

Skeletons (need user to fill model-specific paths / weights):
- `vision-timm/`
- `nlp-hf/`

Templates are read-only references. The run copies them into the run dir, then edits the **copy** — never the source under `assets/templates/`.

## When to load which reference

| File | Load when |
|---|---|
| `references/cv-design.md` | Step 0 — picking the CV scheme |
| `references/template-selection.md` | Step 2 — choosing a template |
| `references/ablation-protocol.md` | Steps 1 / 4 — picking and labeling experiments |
| `references/budget-estimator.md` | Step 3 — estimating wallclock |
| `auto-kaggle/references/state-contract.md` | Always |
| `auto-kaggle/references/integrity-rules.md` | Steps 0, 3, every external data load |
| `auto-kaggle/references/long-running-protocol.md` | Resume / fold-by-fold idempotency |
| `auto-kaggle/references/external-tools.md` | Step 1 — delegating feature brainstorm (handoff #3) or hyperparameter search (handoff #5) |
| `auto-kaggle/references/escalation-policy.md` | Considering CV split change or budget overage |
