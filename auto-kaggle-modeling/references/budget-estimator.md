# Budget estimator

Stage 2 must estimate wallclock for every planned run before starting. Over-budget runs are escalated (Rule 8) — never silently truncated mid-training, which produces useless partial OOFs.

## The formula

```
estimate_hours =
      base_hours_per_fold[task_type]
    × n_folds
    × seed_count
    × data_size_factor
    × cost_multiplier
    × hardware_factor
    × 1.3                          # safety pad
```

| Term | How to compute |
|---|---|
| `base_hours_per_fold[task_type]` | Read from the template's `template.yaml`. Each template ships measured baselines. |
| `n_folds` | From `cv_split.yaml`. |
| `seed_count` | From `config.yaml.seeds` — one fold-suite per seed. |
| `data_size_factor` | `max(1.0, n_train / 1_000_000)` for tabular. For images: `n_train / 100_000`. For NLP: `total_tokens / 100M`. Capped at 5×. |
| `cost_multiplier` | `1.0` for `S`-cost ideas, `1.5` for `M`-cost, `2.5` for `L`-cost. Pick the heaviest idea in `idea_keys`. |
| `hardware_factor` | `1.0` for local-gpu / cloud-gpu. `1.0` for cpu-only on tabular. `0.5` for kaggle-notebook (faster sometimes due to dedicated GPU). `2.0` for cpu-only on anything other than tabular. |
| safety pad | Always 1.3 — we'd rather skip a run than have a budget overrun crash mid-fold 4. |

## Per-template baselines

These are starting points; the agent should refine after observing actual runtimes on the comp's data (write observations into `runs/<comp_slug>/stage2_modeling/budget_observations.md` and use them to update the estimate for future runs).

### `tabular-lgbm`

```yaml
base_hours_per_fold:
  tabular-binary: 0.2      # 1M rows, 50 features, 5000 estimators, early stopping ~1500
  tabular-multiclass: 0.3  # 5 classes
  tabular-regression: 0.2
  tabular-ranking: 0.4
```

Trivially scales with `n_estimators` and `num_leaves`. The `early_stopping_rounds: 200` cap keeps tail variance bounded.

### `vision-timm`

```yaml
base_hours_per_fold:
  image-classification: 2.0    # 50k train images, 384×384, ConvNeXt-base, 10 epochs
  image-segmentation: 4.0
  image-detection: 6.0
  audio-classification: 2.0    # spectrogram input
```

Heavy. Multiply by `data_size_factor` aggressively.

### `nlp-hf`

```yaml
base_hours_per_fold:
  nlp-classification: 1.5      # DeBERTa-v3-base, max_len 512, 50k samples, 3 epochs
  nlp-regression: 1.5
  nlp-token-classification: 2.0
  nlp-generation: 3.0          # seq2seq is slower per epoch
```

For longer sequences (>1024 tokens), multiply by `(max_len / 512) ** 1.6` — attention is super-linear.

## Worked examples

### Tabular regression, 5-fold, 2 seeds, M-cost idea, local-gpu, 2M rows

```
0.2 × 5 × 2 × max(1.0, 2_000_000 / 1_000_000) × 1.5 × 1.0 × 1.3
= 0.2 × 5 × 2 × 2 × 1.5 × 1.3
= 7.8 hours
```

Verify against `compute_env.constraints.max_wallclock_per_run_hours`. If local user said 10h, we're under — proceed. If they said 6h, escalate.

### Vision classification, 5-fold, 1 seed, S-cost, kaggle-notebook, 80k images

```
2.0 × 5 × 1 × max(1.0, 80_000 / 100_000) × 1.0 × 0.5 × 1.3
= 2.0 × 5 × 1 × 1.0 × 0.5 × 1.3
= 6.5 hours
```

Under the 8.5h kaggle-notebook cap → proceed. But this is a single run; if we're doing 4 ablations per day, weekly compute = 4 × 7 × 6.5 = 182h, way over the 30h/week kaggle quota. Stage 2 has a separate weekly-quota check that runs at the start of every cycle.

### NLP classification, max_len 1024, 5-fold, 2 seeds, M-cost, local-gpu, 30k samples

```
1.5 × 5 × 2 × max(1.0, 30_000 × 1024 / 100_000_000) × 1.5 × 1.0 × 1.3
× (1024 / 512) ** 1.6   ← long-sequence penalty

= 1.5 × 5 × 2 × max(1.0, 0.307) × 1.5 × 1.0 × 1.3 × 3.03
= 1.5 × 5 × 2 × 1.0 × 1.5 × 1.0 × 1.3 × 3.03
= 88.6 hours
```

Way over budget. Escalate. Options the agent presents the user:

1. Reduce to 1 seed → 44h (still likely over)
2. Reduce to 3 folds → 27h (within most local-gpu budgets)
3. Use a smaller backbone (DeBERTa-v3-base → small) → ~30h
4. Reduce `max_len` to 512 with sliding-window head → 29h

The agent never picks one autonomously. User decides.

## Weekly compute quota check

For `compute_env.env == kaggle-notebook`:

```python
weekly_quota_h = 30
used_this_week_h = sum_of_run_hours_from_progress_jsonl_since_week_start_utc
if used_this_week_h + estimate_h > weekly_quota_h * 0.9:   # 10% safety margin
    escalate(reason="kaggle-notebook weekly GPU quota would be exceeded",
             options=["skip this run", "wait until next Monday (UTC)",
                      "switch compute env to local/cloud GPU"])
```

For local-gpu / cloud-gpu, weekly quota is by user-supplied policy (lives in `compute_env.yaml`).

## Updating estimates with observed data

After every run completes, append to `budget_observations.md`:

```markdown
- 2026-05-12-lgbm-baseline: estimated 2.6h, actual 1.9h, ratio 0.73 (overestimate)
- 2026-05-12-lgbm-with-log-target: estimated 2.6h, actual 2.1h, ratio 0.81 (overestimate)
```

After 3 observations on the same template / task, the agent computes a `correction_factor = median(actual / estimated)` and multiplies future estimates by it. Bounded to `[0.5, 2.0]` — never assume estimates are off by more than 2×.

## What to do when estimates blow up mid-run

If a run is wall-clocking past 1.5× its estimate and is not done:

1. The training subprocess writes a `BUDGET_WARN` line to `train.log` every 30 minutes past the estimate.
2. The orchestrator polling `train.log` notices, appends `progress.jsonl` event `budget_warn`.
3. At 2× estimate, the orchestrator sends `SIGTERM` to the training subprocess (which is required to save partial OOFs before exiting — they were saved at each `fold_done` event).
4. The run is marked `status: terminated_overbudget` in `leaderboard.csv`. CV is computed on the folds that did complete (so something is salvageable).
5. The budget estimator records this as a 2× underestimate for future correction.

Hard kill at 3× estimate. The user can override with `--no-budget-kill` in extreme cases.
