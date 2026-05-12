# Auto-Kaggle Codex Review

## Critical findings
- `auto-kaggle-modeling/assets/modeling.py:227` — writes relative `data_paths` into `config.yaml`, then runs training with `cwd=run_dir`, so default `--runs-dir runs` makes `pd.read_csv("runs/...")` resolve under the run directory and fail — write resolved absolute paths.
- `auto-kaggle-modeling/assets/modeling.py:329` — invokes `python runs/<slug>/.../train.py` while `cwd=run_dir`, so the script path is double-relative and fails on first default training run — pass `train.py` or `train_py.resolve()`, and pass absolute `--run-dir` / `--progress-log`.
- `auto-kaggle-bootstrap/assets/kaggle_helpers.py:147` — unzips downloaded archives but ignores `unzip` return code, then deletes the zip at `:149`, which can destroy the only downloaded data after a failed unzip — check return code and only unlink after verified extraction.
- `auto-kaggle-bootstrap/assets/bootstrap.py:101` — `is_bootstrap_done()` treats a partial profile as complete after only `comp_profile.yaml` + `hand_off.md`, while the script writes `bootstrap_partial: true` and exits 0 — require `bootstrap_partial: false` plus `rules_summary.md` and `compute_env.yaml` before skipping.
- `auto-kaggle-modeling/assets/templates/ensemble/train.py:202` — copied ensemble runs import `../tabular-lgbm/train.py` from `stage2_modeling/runs/tabular-lgbm`, which will not exist — move shared CV/metric helpers into a copied common module or import from an absolute template source path.

## Important findings
- `auto-kaggle-submit/assets/quota.py:67` — quota parser only counts lines starting with a timestamp, but the repo’s own quota reference shows `fileName date ...`, so used_today can silently become 0 — parse the date column by header/CSV mode or a regex that finds the timestamp after filename.
- `auto-kaggle-submit/assets/submit.py:349` — quota is incremented in memory after submit but never written back to `quota_state.yaml`, and no `quota_used` event is emitted — persist quota state atomically and append `quota_used`.
- `auto-kaggle-submit/assets/submit.py:134` — public-LB polling matches by basename `test_preds.csv`, which is identical for every run, so it can attach an old score to a new submission — submit a unique filename or match by submission id/timestamp/message.
- `auto-kaggle-submit/assets/submit.py:392` — rewrites the last `submission_log.jsonl` line to add public LB, violating the canonical append-only crash contract and risking corruption if another line appears — append a follow-up `public_lb_known` record or use a separate mutable status file.
- `auto-kaggle-modeling/assets/templates/tabular-lgbm/train.py:231` — skipped folds on resume do not reload/recompute fold scores, so `cv_score.json` can be based only on newly trained folds or become 0.0 — persist per-fold score metadata or recompute aggregate CV from full OOF.
- `auto-kaggle-modeling/assets/templates/tabular-lgbm/train.py:188` — multiclass LightGBM sets objective `multiclass` but never sets `num_class` or label-encodes classes — add label encoding and `num_class`, then map outputs to submission columns.
- `auto-kaggle-modeling/assets/templates/tabular-lgbm/train.py:66` — `_split_iter` supports only `StratifiedKFold`, `KFold`, and `GroupKFold`; documented `StratifiedGroupKFold`, `TimeSeriesSplit`, and blocked-time splits fail — implement all documented schemes or reject them before training.
- `auto-kaggle-submit/assets/near_duplicate.py:40` — near-duplicate checking sorts by one id column and converts every non-id column to float, which breaks event detection, detection, string-label, and multi-key submissions — use `comp_profile.submission.columns` to identify key and prediction columns.
- `auto-kaggle-submit/assets/recommend.py:146` — recommendations can put runs with missing `test_preds.csv` or no valid attribution into top slots — filter eligibility using the same pre-submit gates as `submit.py`.
- `auto-kaggle-bootstrap/assets/kaggle_helpers.py:176` — attribution enforcement only checks for substring `attr:` or `Derived from:` and does not validate `author/kernel` tokens — validate the required `attr: <author>/<kernel-slug>` pattern or define an explicit own-only syntax.

## Contract drift
- `auto-kaggle/references/state-contract.md:101` → `auto-kaggle-bootstrap/references/task-type-detection.md:5` / templates → canonical task types disagree (`binary`, `tabular-classification`, `object-detection` vs `tabular-binary`, `tabular-multiclass`, `image-detection`).
- `auto-kaggle-bootstrap/SKILL.md:30` → `auto-kaggle-bootstrap/assets/bootstrap.py:184` → SKILL promises final `rules_summary.md`, `compute_env.yaml`, and complete `comp_profile.yaml`; script writes a partial profile and no rules or compute env.
- `auto-kaggle/references/long-running-protocol.md:39` → `auto-kaggle-bootstrap/assets/bootstrap.py:224` → documented `comp_profile_written` / `bootstrap_done` events are not emitted.
- `auto-kaggle-modeling/SKILL.md:206` → `auto-kaggle-modeling/assets/modeling.py:357` → Stage 2 never writes `stage2_modeling/hand_off.md`.
- `auto-kaggle-submit/SKILL.md:213` → `auto-kaggle-submit/assets/submit.py:388` → Stage 3 never writes `stage3_submit/hand_off.md`.
- `auto-kaggle-submit/SKILL.md:239` → `auto-kaggle-submit/assets/recommend.py:135` / `submit.py:288` → deadline mode does not write `final_selection.md`, rename final slots, reserve quota, or enforce the 24h workflow.
- `auto-kaggle/references/integrity-rules.md:76` → Stage 1/2 code → external-data approval is documented but not enforced; no code checks `external_data.yaml` or `external_data_candidates.md` before training.
- `auto-kaggle-modeling/references/budget-estimator.md:7` → `auto-kaggle-modeling/assets/modeling.py:122` → estimator ignores data size, idea cost, hardware factor, long-sequence penalty, weekly Kaggle quota, and observed correction factors.
- `auto-kaggle/references/long-running-protocol.md:77` → `auto-kaggle/assets/supervisor.sh:136` → supervisor sleeps the whole orchestrator while `wait_until.txt` exists, preventing allowed non-submission recon/modeling work.

## Cross-script consistency
- `auto-kaggle-modeling/assets/templates/vision-timm/train.py:137`, `vision-timm-seg/train.py:177`, `vision-det/train.py:126`, `nlp-hf/train.py:120`, `ensemble/train.py:202` — all copied templates use the same broken relative import for tabular helpers.
- `auto-kaggle-submit/assets/submit.py:307` → `auto-kaggle-bootstrap/assets/kaggle_helpers.py:178` — submit.py allows own-only runs via `has_own`, but `submit_csv()` rejects messages without `attr:`.
- `auto-kaggle-submit/assets/quota.py:67` → `auto-kaggle-bootstrap/assets/kaggle_helpers.py:250` — two quota parsers exist with the same date-at-start assumption; centralize one parser and test it against saved CLI output.
- `auto-kaggle-modeling/assets/leaderboard.py:132` → `auto-kaggle-submit/assets/recommend.py:60` — leaderboard can store `public_lb`, but recommendations only trust `submission_log.jsonl`, so LB data can disappear after log parse issues.
- `auto-kaggle-submit/assets/submit.py:318` → `kaggle_helpers.py:189` and `submit.py:347` — one Kaggle submit produces two `submitted` progress events.

## Minor
- `auto-kaggle-bootstrap/assets/bootstrap.py:145` — writes plain text to `raw_comp_view.json`; use `.txt` or actual JSON if available.
- `auto-kaggle-submit/assets/submit.py:39` — `shlex` and `subprocess` are imported but unused.
- `auto-kaggle-modeling/assets/leaderboard.py:120` — `record_run_failed()` writes `<run_id>.err` non-atomically.
- Kaggle CLI was not installed in this environment, so I could not verify live `kaggle --help` output or actual table formats locally.

## Spot-checks performed
- Read carefully: all `auto-kaggle*/assets/**/*.py`, all five `SKILL.md` files, all `auto-kaggle*/references/*.md`, and `auto-kaggle/assets/supervisor.sh`.
- Ran a syntax-only AST parse over `auto-kaggle*/assets/**/*.py`; no Python syntax errors found.
- Did not run networked Kaggle commands; attempted local `kaggle ... --help`, but `kaggle` was not installed.