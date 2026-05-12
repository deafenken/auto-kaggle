# tabular-lgbm template

LightGBM training entry point for tabular binary / multiclass / regression / ranking tasks.

Invoked by `auto-kaggle-modeling/assets/modeling.py` after it copies this directory into a run dir at `runs/<comp_slug>/stage2_modeling/runs/<run_id>/`.

## Contract this template implements

- Reads `cv_split.yaml` from `runs/<comp_slug>/stage2_modeling/cv_split.yaml`.
- Trains one OOF per fold per seed.
- After every fold completes, atomically saves the fold's OOF + test prediction slices, then writes a `fold_done` event to `progress.jsonl` (passed in via `--progress-log`).
- On a crash + resume, skips folds whose slice files exist with a matching config hash.
- On all-folds-done, writes:
  - `oof.npy` — full out-of-fold predictions on the train set
  - `test_preds.csv` — predictions on the test set in submission format
  - `cv_score.json` — metric / per_fold / mean / std

## Config knobs (see `template.yaml` for defaults)

| Key | Default | Notes |
|---|---|---|
| `device` | `cpu` | `gpu` if compute_env has CUDA |
| `n_estimators` | 5000 | hard cap; early stopping usually kicks in well before |
| `early_stopping_rounds` | 200 | per LightGBM |
| `num_leaves` | 63 | doubling this roughly doubles train time |
| `learning_rate` | 0.02 | with 200 early-stopping rounds, this hits convergence around iter 1500–3000 |
| `feature_fraction` | 0.9 | sampled per iteration |
| `bagging_fraction` | 0.85 | with `bagging_freq` |
| `min_data_in_leaf` | 100 | larger → less overfit, less expressive |
| `feature_config.target_transform` | null | `log1p` / `boxcox` if target is skewed |
| `feature_config.drop_cols` | [] | columns to skip entirely |
| `feature_config.target_col` | `target` | from comp_profile |
| `feature_config.id_col` | `id` | from comp_profile |
| `seeds` | [42] | one OOF + test set of slices per seed; final preds averaged |

## What this template does NOT do

- Feature engineering beyond category dtype handling. Add features by editing `pipeline.py` in the run dir OR by pre-computing them in `data/processed/`.
- Hyperparameter search. Use Optuna externally (see `auto-kaggle/references/external-tools.md` handoff #5).
- Ensembling. That's a separate run with `template: ensemble` (planned).
- Non-Kaggle export. Predictions go into `sample_submission.csv`'s schema verbatim.

## Editing the copy, not the source

The agent edits `runs/.../<run_id>/train.py` (the copy), not this file. Source-edits affect future runs; copy-edits affect only this run. If you find yourself editing the source for a specific competition, that's a signal you should either (a) parameterize it via config, or (b) make a fork like `tabular-lgbm-with-target-encoding/`.
