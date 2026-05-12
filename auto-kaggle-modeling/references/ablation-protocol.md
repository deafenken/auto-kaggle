# Ablation protocol

Every Stage 2 run is an ablation: it changes **one thing** (or a tightly coupled set) relative to the prior best run, so the resulting CV delta is attributable to that change. This is what separates "we got lucky" from "this technique works."

## The one-thing-at-a-time rule

A new run's `config.yaml` should differ from the prior best run's `config.yaml` in **one** of:

- A new idea key in `idea_keys` (one new technique added).
- A removed idea key (we're checking whether an idea actually helps by running without it).
- A change in seed (variance estimation, not strictly an ablation).
- A change in `n_folds` (CV-noise estimation).
- An ensemble step on top of existing OOFs (this is an aggregation, not a single-model change).

If two things change at once and CV improves, you don't know which one helped. That's not an ablation, it's a guess.

Exception: when the new idea **depends on** another (e.g. "fit a meta-learner on OOFs" requires the OOFs to exist). In that case the dependency was set up in a prior run and counts as part of the existing pipeline.

## Pipeline.py vs config-driven ablation

Two kinds of "one thing" changes:

### Config-driven (preferred)

The change can be expressed in `config.yaml`. The same `pipeline.py` / `train.py` is used. This is the cheapest kind of ablation:

```yaml
# Prior best run config.yaml
idea_keys: [cv:stratified-kfold-5, feature:log1p_target_then_xgboost]
model_params:
  num_leaves: 63

# New run config.yaml (ablation)
idea_keys: [cv:stratified-kfold-5, feature:log1p_target_then_xgboost, aug:mixup-tabular]
model_params:
  num_leaves: 63
augmentation:                     # new section
  type: mixup-tabular
  alpha: 0.2
```

### Pipeline-driven (necessary when the change is structural)

The change can't be expressed in `config.yaml` — it requires editing `pipeline.py`. E.g. a new feature requires new code; a new model requires a new template.

When this happens:

1. The new run gets a fresh `pipeline.py` (copied from the template, then modified).
2. `attribution.md` notes which functions in `pipeline.py` are new and cites the source kernel.
3. The diff is committed conceptually to `attribution.md`; we don't actually `git diff` since `runs/` is gitignored.

## Attribution per run

Every run dir has `attribution.md`:

```markdown
# Attribution — 2026-05-12-lgbm-with-log-target

## Ideas used (from ideas_pool.md)
- `cv:stratified-kfold-5`           — cite: jdoe123/eda-and-lgbm-baseline, msmith/lgbm-tricks
- `feature:log1p_target_then_xgboost` — cite: jdoe123/eda-and-lgbm-baseline, kdawg/regression-tweaks

## Changes vs prior best (2026-05-12-lgbm-baseline)
- ADDED: `feature:log1p_target_then_xgboost`
- Code change: `pipeline.py` lines 87–94 — added `apply_target_transform` function

## Own additions (not from any kernel)
- None

## Why this should help here
<one paragraph explaining the hypothesis>

## After-run notes (filled in by agent after CV is in)
- CV result: <metric> = <score> (std <std>) — <improved | regressed | within noise>
- Delta vs prior best: <+/- delta> (<+/- in stds>)
- Verdict: <"keep — significant improvement" | "drop — within noise" | "investigate further">
```

If `Own additions` is non-empty (you came up with something not from any kernel), that's fine — but it changes the attribution rule for Stage 3. The submission message gets an extra `+own` token: `"... | attr: ... +own"`.

## Verdict thresholds

After a run completes:

| CV delta | Verdict |
|---|---|
| > +1 × prior cv_std | "keep — significant improvement" |
| +0.5 to +1 × prior cv_std | "keep but marginal — needs another seed to confirm" |
| -0.5 to +0.5 × prior cv_std | "within noise — neither keep nor drop" |
| -0.5 to -1 × prior cv_std | "regression — drop, but reconsider in ensemble" |
| < -1 × prior cv_std | "drop — significant regression" |

"Within noise" results still contribute to the OOF pool for ensembling later — diversity has value even when single-model CV is unmoved.

## Running an "ablation away" check

Periodically (every 10 runs, or when CV plateaus for 5 consecutive runs), pick the run that contributed the smallest CV delta and run an "ablation away" — same config but without that idea. If CV doesn't drop, that idea is dead weight and should be dropped from the cumulative pipeline.

This is how the cumulative pipeline stays clean. Without it, you accumulate "ideas that we tried, that didn't hurt, that we kept" and the pipeline gets noisier over time.

## What `idea_keys` does and doesn't track

`idea_keys` in `config.yaml` is the list of `ideas_pool.md` entries this run is implementing. It does **not** track:

- Standard pipeline elements (data loading, basic preprocessing, CV setup) — those are always present.
- Template defaults — e.g. LightGBM with `num_leaves=63` doesn't get an idea key.
- "We tried it and it didn't work" — those are recorded in `attribution.md` as "tried but reverted", not in `idea_keys`.

`idea_keys` is the menu items, not the kitchen procedure.

## Multi-seed runs are not ablations

Running the same config with `seed: 1337` vs `seed: 42` is a variance estimation, not an ablation. Both runs share the same `idea_keys` and the same attribution. Use them to update `cv_std` more accurately. The `run_id` distinguishes them: `2026-05-12-lgbm-baseline-s42` vs `2026-05-12-lgbm-baseline-s1337`.

When you average their OOFs and test predictions to produce a single "blended" entry, that's a new ensemble run with a distinct `run_id`, `template: ensemble`, and `idea_keys: [ensemble:multi-seed-average]`.

## Ablation order matters less than you think

You do not need to find the globally-best order to try ideas. Greedy priority (Step 1 in `SKILL.md`) is good enough. What you must NOT do:

- Stack too many ideas in one run (you lose attribution).
- Skip the priority queue to chase a hunch (you starve the queue).
- Forget to update `attribution.md` (Stage 3 will refuse to submit a run with empty / missing attribution).

The queue is the spine of the run — follow it, log results, repeat.
