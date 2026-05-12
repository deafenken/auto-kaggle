# ensemble template

Blends or stacks the OOFs / test preds of prior `stage2_modeling/runs/*/` runs to produce a new candidate.

The ensemble is itself a Stage 2 run (same contract: cv_score.json + test_preds.csv + attribution.md + oof.npy). Stage 3 ranks it alongside single-model runs.

## When to use

- **`improvement mode`**: when `ideas_pool.md` is exhausted and `untried` is empty. Pick the top 3–5 runs by trust-adjusted CV, blend them.
- **Multi-seed average**: same architecture, different seeds → ensemble template with arithmetic blend.
- **Diversification pre-deadline**: blend models from different families (LightGBM + CatBoost + a small neural net).

## Modes

### `blend` (default — start here)

```yaml
mode: blend
source_runs:
  - 2026-05-12-lgbm-baseline-s42
  - 2026-05-12-lgbm-baseline-s1337
  - 2026-05-12-cat-baseline
blend:
  aggregation: arithmetic   # arithmetic | geometric | rank_mean
  weights: null             # null = uniform
```

**Aggregation choices:**
- `arithmetic` — simple average. Always safe.
- `geometric` — `exp(mean(log(preds)))`. Use when targets are probabilities (lives natively in log-odds space).
- `rank_mean` — rank-normalize each source's predictions to [0, 1], then average. Useful when sources have wildly different output scales (e.g. blending logits with probabilities) or when the metric is rank-based (AUC, NDCG).

**Weights:** start with `null` (uniform). If one source is much weaker, downweight or drop it instead of fitting weights — fitted weights on a tiny CV signal overfit fast.

### `stack`

```yaml
mode: stack
source_runs: [...]
stack:
  meta_model: ridge        # ridge | lgbm
  ridge_alpha: 1.0
  lgbm_params: {num_leaves: 7, learning_rate: 0.05, n_estimators: 500}
```

Fits a meta-learner on the OOF matrix (sources × n_train) using the SAME CV scheme as the source runs. Default meta-model is `ridge` because it's the safest — `lgbm` can overfit dramatically on the small "OOF as features" matrix.

Use `stack` when:
- You have ≥ 4 reasonably diverse source runs (each contributes signal beyond what the others have).
- Source runs' raw CV is similar; you want the meta-learner to figure out the optimal weighting.

Avoid `stack` when:
- You have fewer than 3 sources (just blend).
- One source is clearly best; stacking will likely downweight it.
- The CV scheme is time-series; stacking on TimeSeriesSplit needs extra care (the meta-learner sees only past folds' OOFs at each timepoint).

## Config knobs

| Key | Default | Notes |
|---|---|---|
| `mode` | `blend` | `blend` or `stack` |
| `source_runs` | `[]` | required; must point to runs with completed `oof.npy` and `test_preds.csv` |
| `blend.aggregation` | `arithmetic` | choose by metric (rank for AUC/NDCG, geometric for probability blends) |
| `blend.weights` | `null` | uniform if null |
| `blend.clip_to_unit` | `false` | clip outputs to [0, 1] for probability outputs |
| `stack.meta_model` | `ridge` | `ridge` (safe) or `lgbm` (riskier, more expressive) |
| `stack.ridge_alpha` | `1.0` | higher = more regularization; tune in [0.1, 10] |

## Source run requirements

For every `run_id` in `source_runs`:

- `runs/<comp>/stage2_modeling/runs/<run_id>/oof.npy` must exist and be shape `(n_train,)` or `(n_train, n_classes)`.
- `runs/<comp>/stage2_modeling/runs/<run_id>/test_preds.csv` must exist and have the same id column as the comp's `sample_submission.csv`.
- `runs/<comp>/stage2_modeling/runs/<run_id>/cv_score.json` must exist (used to inform you which CV the source had).

If any source is missing, the script aborts with a clear error — better than silently dropping a source.

## Compute cost

Effectively zero compared to training a model from scratch. Reads N × 100K-ish-row arrays from disk, does linear algebra, writes outputs. Runs in seconds for tabular, minutes for vision.

## Attribution

`modeling.py` pre-writes `attribution.md` listing the source runs and any kernels from `ideas_pool.md` cited as `ensemble:*` techniques. This template appends an "Ensemble construction" section recording the actual `mode`, `aggregation` / `meta_model`, and combined CV.

The Stage 3 submission message will carry `attr:` for the source kernels via Stage 2's normal attribution propagation. Add `ensemble:<source_count>-blend` or `ensemble:stack-ridge` as an idea key in `idea_keys` so it shows up in `recommendations.md` next to the run.

## What this template does NOT do

- Fit blend weights via numerical optimization (e.g. NNLS on OOFs). This frequently overfits on small CV signal — `stack` with Ridge is a safer route to "learn the weights".
- Calibrate the output distribution. Use a separate post-processing run (`post:` category) for that.
- Mix multi-class probabilities with regression targets. Source runs must all have the same output shape.
