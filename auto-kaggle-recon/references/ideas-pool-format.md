# `ideas_pool.md` format

The output of distillation. Read by Stage 2 to decide which ideas to try next, and by Stage 3 to write submission attribution lines. The file is **only ever appended to or edited in-place** — never wholesale-rewritten, so partial distillation across resumes is safe.

## Top-level layout

```markdown
# Ideas pool — <comp_slug>

_Updated <ISO-8601 UTC>, last recon cycle <N>, <total> ideas across <K> kernels._

## Consensus (≥3 citations from top-10)
- … (links to entries below)

## New this cycle
- …

---

## CV scheme: StratifiedKFold n=5 shuffle=True seed=42
…

## Feature: target_log_transform
…
```

A "Consensus" and "New this cycle" header at the top give Stage 2 the two highest-priority lists at a glance.

## Per-idea entry schema

Each idea is a level-2 heading (`## <category>: <short name>`) followed by a fixed-key block:

```markdown
## feature: log1p_target_then_xgboost

**Category:** feature                    (one of: cv | feature | model | aug | ensemble | post | external_data)
**Cost:** S                              (S / M / L — see "Cost estimation" below)
**Effect (reported):** +0.0031 CV RMSE   (verbatim from kernel if stated; "n/a" if not)
**Consensus:** true (5 kernels)          (true if ≥3 top-10 kernels use it, with count)
**Composes with:** [model:catboost-with-cat-cols, ensemble:lgbm-cat-blend]
**Conflicts with:** []
**Citations:** [jdoe123/eda-and-lgbm-baseline, msmith/feature-engineering-walkthrough,
                kdawg/lgbm-target-tricks, vchen/regression-baselines, alopez/yet-another-blend]
**Distilled at (UTC):** 2026-05-12T10:32:00Z

### Description
Apply `np.log1p` to the regression target before training, predict in log-space, then `np.expm1`
the predictions before computing the metric. Justified because the target is right-skewed
(visible in stage0_bootstrap/data_stats.md) and the metric is RMSE — log-space MSE is closer
to RMSLE which down-weights large outliers.

### Why it might help here
- The target has long-tail outliers (top 0.1% are ~10× the median).
- Several baselines report consistent +0.002 to +0.005 CV from the transform.
- Zero risk of overfitting (deterministic transform).

### How to implement (notes, not code)
- Wrap the target column in `log1p` inside the data loader, not in the training loop, so the
  CV split sees the transformed target consistently.
- Remember `expm1` before computing the competition metric and before writing test predictions.
- For multi-target settings, apply per column based on each target's skew.

### Risks / gotchas
- If the target has negative values, `log1p` is not defined — use a shifted version `log(target + C)`
  with `C` chosen so the minimum is >= 1.
- Stacking with an in-target-space model needs un-transformation before stacking.
```

## Field rules

- **Category:** must be one of the 7 fixed strings. The Stage 2 priority rule uses category to balance the experiment portfolio (no running 5 ensemble experiments in a row while ignoring features).
- **Cost:** `S` (< 30 min wallclock to implement and run a single fold), `M` (30 min – 2h), `L` (> 2h or requires custom code). Stage 2 budget estimator multiplies by fold count.
- **Effect (reported):** verbatim if the kernel author quoted a number. Never extrapolate — if you have to estimate, write `"reported as `<>` by <ref>, others silent"`.
- **Consensus:** `true` if ≥3 kernels in the current top-10 by votes or by public LB cite it. The count in parens is over the full citations list, not just top-10.
- **Composes with / Conflicts with:** references to other entries' short-names. Empty lists `[]` are fine. The agent fills these as it distills more kernels and notices conflicts (e.g. "target log transform" conflicts with "Tweedie loss" because both reshape the target distribution).
- **Citations:** at least 1; use the `ref` strings from `kernels_index.json`. These become the `attr:` tokens in submission messages.
- **Distilled at:** ISO-8601 UTC; lets the agent skip re-distillation.

## Dedup protocol

When distilling a new kernel:

1. For each technique seen in the kernel, look for an existing entry under the matching `## category: <name>` with the same conceptual content (semantic match — agent's judgment, not string match).
2. If a match exists: append this kernel's `ref` to that entry's Citations list (deduped), recompute Consensus, possibly tighten the Description if the new kernel adds detail.
3. If no match: write a new entry at the bottom of the file under the appropriate category section.

Never duplicate entries. If you find duplicates after the fact (the agent merged ideas under different names), merge them in a single edit and append a `> Merged with: <other-name>` line at the bottom of the surviving entry.

## "Consensus" and "New this cycle" lists at the top

After each recon cycle finishes, the agent regenerates the two summary lists at the top:

```markdown
## Consensus (≥3 citations from top-10)
- [feature:log1p_target_then_xgboost](#feature-log1p_target_then_xgboost) — 5 kernels
- [cv:stratified-kfold-5](#cv-stratified-kfold-5) — 9 kernels
- [model:catboost-with-cat-cols](#model-catboost-with-cat-cols) — 4 kernels

## New this cycle
- [feature:date-cycle-encoding](#feature-date-cycle-encoding) — added 2026-05-12T10:32Z
- [aug:mixup-tabular](#aug-mixup-tabular) — added 2026-05-12T10:34Z
```

Stage 2's priority rule reads these first. The Consensus list is "try these soon"; the New this cycle list is "consider these in the next experiment slot."

## What does **not** belong in `ideas_pool.md`

- Code blocks copied from a kernel. Rule 1 — re-implementation lives in `stage2_modeling/pipeline.py`, never here.
- Random thoughts from the agent that are not grounded in a kernel. If an idea has no citation, do not include it. (Brainstorming goes to `stage2_modeling/feature_ideas.md` instead — see external-tools handoff #3.)
- Speculations about private LB ("this might shake up the LB"). Speculation goes into Stage 2 attribution notes if it shapes a run, not into the idea pool.
- Anything that contradicts `comp_profile.yaml` (e.g. "use external dataset X" when external data is forbidden — that goes into `external_data_candidates.md` and gets flagged for user review, but is not added as a usable idea).
