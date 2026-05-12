# CV scheme design

Decided once per competition at the start of Stage 2. Written into `cv_split.yaml` with rationale. **Only the user changes it after that** — Rule 7.

This file is the decision tree.

## Why CV is the single most important file

CV is what predicts your private LB. If your CV says +0.005 from a change and public LB says –0.001, **trust CV**. If your CV says –0.003 and public LB says +0.012, **still trust CV** — that's a public-LB overfit, and shake-up is going to eat it.

A bad CV (one that doesn't reflect the test distribution) makes the entire pipeline produce shake-up casualties. We pick CV first, then accept whatever it tells us thereafter.

## Decision tree

```
                  ┌──────────────────────────┐
                  │ comp_profile.task_type   │
                  └──────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────┐
                  │ time-series-* ?         │── yes ──► TimeSeriesSplit OR blocked-split (§ 4)
                  └─────────────────────────┘
                                │ no
                                ▼
                  ┌─────────────────────────┐
                  │ Detected group column?  │── yes ──► GroupKFold (§ 2)
                  └─────────────────────────┘
                                │ no
                                ▼
                  ┌─────────────────────────┐
                  │ Classification?         │── yes ──► StratifiedKFold (§ 1)
                  └─────────────────────────┘
                                │ no (regression)
                                ▼
                  ┌─────────────────────────┐
                  │ Long-tail target?       │── yes ──► StratifiedKFold on binned target (§ 1b)
                  └─────────────────────────┘
                                │ no
                                ▼
                            KFold (§ 1c)
```

## 1. StratifiedKFold (classification)

```yaml
scheme: StratifiedKFold
n_folds: 5
shuffle: true
seed: 42
stratify_col: target
rationale: |
  Binary / multiclass classification with no group structure. Stratify on the
  target column to preserve class balance across folds — critical when one class
  is < 20% of the data. Seed 42 fixed for reproducibility.
```

`n_folds` defaults to 5. Going to 10 doubles training cost; only worth it if you have compute headroom AND the CV signal is noisy (cv_std > 0.5 × cv_mean is a typical threshold).

### 1b. StratifiedKFold on binned regression target

When the target is regression but heavily skewed:

```yaml
scheme: StratifiedKFold
n_folds: 5
shuffle: true
seed: 42
stratify_col: target_bin  # 10 quantile bins, computed at split time
rationale: |
  Regression with right-skewed target (skewness > 1.5 in data_stats.md).
  Stratify on 10 quantile bins to ensure every fold sees the long tail.
```

### 1c. KFold (regression, no skew, no group)

```yaml
scheme: KFold
n_folds: 5
shuffle: true
seed: 42
rationale: |
  Regression with approximately symmetric target distribution and no group
  structure. Plain KFold with shuffle is sufficient.
```

## 2. GroupKFold

Use when a single semantic "group" appears across multiple rows and the test set has *different* groups (the classic CV-LB-mismatch trap).

Common group columns:

| Task | Likely group col |
|---|---|
| Patient health prediction | `patient_id` / `subject_id` |
| Multi-image-per-instance | `series_id` / `case_id` |
| Time-series-by-entity | `entity_id` / `series_id` |
| Multi-question per user | `user_id` |
| Multi-segment per text | `document_id` |

Detect by checking `data_stats.md` for columns with high cardinality that appear in both train and `sample_submission`'s id structure. If the group column is unclear, escalate — guessing wrong silently is worse than asking.

```yaml
scheme: GroupKFold
n_folds: 5
group_col: subject_id
rationale: |
  Each subject contributes multiple rows. The test set contains subjects NOT
  in train (verified by checking subject_id overlap = 0). GroupKFold prevents
  the model from "memorizing" subjects and inflating CV.
seed: null   # GroupKFold has no seed; folds are deterministic given the group col
```

If you can't verify the train/test subject overlap (because test labels are hidden), assume the comp organizer used a held-out group split — they almost always do.

## 3. Stratified GroupKFold

When you have both: a group column AND a need to stratify on target.

```yaml
scheme: StratifiedGroupKFold
n_folds: 5
group_col: patient_id
stratify_col: target
rationale: |
  Multiple measurements per patient; rare class makes up < 5% of subjects. We need
  to ensure every fold has both the rare class AND held-out patients.
```

`sklearn.model_selection.StratifiedGroupKFold` since version 1.0.

## 4. Time-series splits

The default for any task with a temporal dimension.

### 4a. TimeSeriesSplit (forecasting)

```yaml
scheme: TimeSeriesSplit
n_folds: 5
gap_days: 7
test_size_days: 30
rationale: |
  Forecasting with a continuous timeline. Each fold trains on data up to T_i
  and validates on the next 30 days. A 7-day gap prevents the validation
  fold from leaking via lag features computed at training time.
```

The `gap_days` is critical and frequently missed. If your features include rolling means or lags, the rolling window at training-end can spill into the validation start. The gap kills that spill.

### 4b. Blocked time split (custom)

For competitions where time has natural blocks (years, seasons):

```yaml
scheme: blocked-time
n_folds: 5
blocks:
  - train: ["2019", "2020", "2021"]
    val:   ["2022"]
  - train: ["2020", "2021", "2022"]
    val:   ["2023"]
  # ...
rationale: |
  Annual blocks. The test set is presumed to be 2024. Each fold validates on
  a year held out from training, preserving long-range temporal structure.
```

Document the block structure verbatim; an agent reading this later needs to recreate the split exactly.

### 4c. Time-series event detection

A special case: you predict per-timestamp events (like sleep stage transitions). CV here is by **series** (whole time series held out), with `GroupKFold` on `series_id`. Same as §2.

## Common mistakes that wreck a CV scheme

1. **Forgetting to seed when shuffle=True.** Folds change every run; ablations become meaningless. Always set `seed`.
2. **Picking n_folds based on superstition.** 5 is fine. 10 is fine. 3 is too few. >10 is wasteful.
3. **Re-tuning the split after seeing the LB.** Rule 7 — DON'T. If CV-LB disagrees, the right response is "investigate the gap" not "make CV match LB."
4. **GroupKFold without verifying the group is what you think it is.** Run a quick `train.groupby(group_col).size()` and confirm the cardinality is reasonable (not 1 group with 99% of data).
5. **TimeSeriesSplit on shuffled data.** The whole point is preserving time order. Don't `shuffle=True`.

## Validating the split before committing

Before writing `cv_split.yaml`, do this sanity check in code:

```python
from sklearn.model_selection import ...

splitter = make_splitter(scheme, ...)
folds = list(splitter.split(X, y, groups=groups if has_groups else None))

# Check each fold:
for i, (tr, va) in enumerate(folds):
    print(f"Fold {i}: train={len(tr)}, val={len(va)}")
    if has_target:
        print(f"  train target mean: {y[tr].mean():.4f}, val: {y[va].mean():.4f}")
    if has_groups:
        train_groups = set(groups[tr])
        val_groups = set(groups[va])
        overlap = train_groups & val_groups
        assert not overlap, f"Group overlap in fold {i}: {len(overlap)}"
```

Group-overlap assertion failing is a stop-the-world bug. Fix the split before training a single model.

## CV-LB gap and what it means

After the first run, you have one data point: `cv_score` and `public_lb`. The "gap" is `|cv - lb|`. Use it as a sanity check, not a tuning signal.

| Gap | Read |
|---|---|
| < 1 × cv_std | Healthy; CV is a good predictor of LB |
| 1–2 × cv_std | Mild discrepancy; usually fine but flag in `hand_off.md` |
| 2–4 × cv_std | Something is off — investigate (leakage? distribution shift? noisy LB?) — flag soft escalation |
| > 4 × cv_std | Hard escalation — likely a CV scheme problem, not a model problem |

The investigation never modifies `cv_split.yaml` autonomously. The agent escalates with diagnostics and the user decides.
