# Kernel distillation — turning a notebook into ideas

The hardest part of Stage 1. The CLI gives us source files; the agent has to read them and emit structured ideas. This file is the reading guide.

## What "distillation" means in this pipeline

Reading a Kaggle notebook (or script) and writing one or more entries into `ideas_pool.md` that capture, in natural language, the techniques the kernel author used. **The output is descriptions, not code.** Rule 1 — code lifting is forbidden.

Goal: a Stage 2 agent that has never read the kernel can re-implement the technique from the description alone.

## Reading order inside a kernel folder

A kernel pulled with `-m` produces:

```
runs/<slug>/stage1_recon/kernels/<author>__<slug>/
├── <slug>.ipynb            # the notebook (or .py for scripts)
├── kernel-metadata.json    # title, language, runtime, dataSources, license
└── (sometimes) requirements.txt, output files
```

Read in this order:

1. **`kernel-metadata.json` first.** Note `dataSources` (external datasets the kernel depends on — those go into `external_data_candidates.md`), `language` (Python/R), and `kernelType` (Notebook/Script). License determines what citation note to use (Apache-2.0, CC BY-SA 4.0, MIT, proprietary).
2. **The notebook's first markdown cell.** Authors often state their approach upfront ("LightGBM with 5-fold stratified KFold + 12 engineered features").
3. **All markdown cells.** Treat them as the author's narration. They explain *why* the author chose a technique — gold for the "Why it might help here" section of an idea entry.
4. **Code cells, scanned for technique-level patterns** (next section). Do not read line-by-line; you are looking for the cookbook, not the recipe.
5. **Output cells**, if present. CV scores and ablation tables go into the idea's "Effect (reported)" field if quoted.

## Technique-level patterns to detect

For each category, here are the patterns the agent should grep for. Treat this list as a primer — actual technique detection is the agent's judgment.

### `cv` — cross-validation scheme

- `StratifiedKFold(n=...)` → CV scheme entry with the n value
- `KFold(n=..., shuffle=..., seed=...)` → likewise
- `GroupKFold` / custom group-aware splitting → record the group column
- `TimeSeriesSplit` / manual time-aware split → time-aware CV entry
- Multi-seed CV ("average 5 seeds of 5 folds = 25 trains") → consensus indicator
- Stratification by a derived column (e.g. binned target) → variant entry

### `feature` — feature engineering

- New columns derived from `train.csv` / `test.csv`:
  - aggregations (`groupby(...).mean()`, `transform`) → entry per aggregation family
  - date decompositions (`dt.year`, cyclical sin/cos encoding) → entry
  - text features (`tfidf`, `count_vectorize`, `len(text)`) → entry
  - target encoding / Bayesian smoothing → entry (and flag as leak-risk in description)
  - interactions / polynomial features → entry
- Target transforms (`log1p`, `boxcox`, `winsorize`) → entry under `feature`
- Categorical handling: label encode / ordinal / one-hot / target-encoded → entry

### `model` — model architecture

- Specific model + key hyperparameters that are unusual ("LightGBM with `num_leaves=255, learning_rate=0.005, max_bin=256`")
- Non-default loss functions (`tweedie`, `quantile`, `focal`) → entry
- Pre-trained backbones (`timm.create_model('eva02_large_patch14_clip_336', pretrained=True)`)
- Customized heads (`nn.Sequential(...)`) or pooling layers
- Sequence-aware models (LSTM stacks, Conformer-style)

### `aug` — augmentation

- For tabular: `mixup`, `cutmix-tabular`, label smoothing, noise injection
- For vision: rotation / flip / color jitter / CutMix / MixUp / RandAugment / RandomErasing; the **policy** is the entry, not each operation
- For NLP: back-translation, synonym replacement, prompt prefixing, masked-token shuffling
- For time series: jittering, time warping, magnitude scaling

### `ensemble` — combining models

- Simple averaging (arithmetic / geometric / rank-mean) → entry, note which
- Weighted blend with weights learned via OOF → entry
- Stacking: meta-learner (Ridge / LGBM) on OOF predictions → entry
- Pseudo-labeling: train on test set predictions, often iteratively → entry (flag overfit risk)
- TTA (test-time augmentation) → entry under `ensemble`

### `post` — post-processing

- Calibration: isotonic / Platt scaling
- Thresholding for classification (per-class optimal threshold via OOF)
- Snapping predictions to legal values (integer rounding, clipping to [0, 1])
- Combining sub-predictions per row (e.g. detection NMS variants)

### `external_data` — extra training data

- Any `pd.read_csv` or `kaggle datasets download` of a non-competition dataset
- Pretrained model weights pulled from outside the comp's dataset
- These also go into `external_data_candidates.md` for user approval

## Cost estimation (`S` / `M` / `L`)

Make this fast — it's a rough triage, not a project plan.

| Cost | Implementation time | Single-fold runtime |
|---|---|---|
| `S` | < 30 min | minimal — adds nothing or seconds to existing training |
| `M` | 30 min – 2 h | doubles training time or requires new dependencies |
| `L` | > 2 h | requires custom kernels, multi-stage training, or non-trivial CI |

Cost is *per-idea-when-added-on-top-of-current-pipeline*, not absolute. A new feature is `S` if the pipeline already has a feature pipeline; `M` if you have to build that pipeline first.

## What **not** to extract

- The kernel's specific hyperparameter values, unless they are unusual and clearly motivated. (LightGBM `num_leaves=31` is the default and not worth recording.)
- The kernel's data loading boilerplate. We have our own.
- Anything that uses prohibited external data (cross-check against `comp_profile.external_data.allowed` — if false, do **not** extract that technique as a usable idea).
- Code comments and docstrings — they go into the idea's "Description" only if they explain *why* the technique exists, not what it does.

## Edge cases

- **Kernel is a clone of another with minor tweaks.** Detect by similar text + same author within a short window. Cite the original; do not create a separate set of entries.
- **Kernel is an unrelated re-skin.** Detect by mismatched feature names or unrelated metric. Mark the kernel in the index as `off_topic: true` and skip.
- **Kernel uses techniques explicitly forbidden by the rules.** This happens — flag `rule_violation_suspect: true` in the index, log it for the user, do not extract those techniques. Do not assume the kernel author knew the rule; just record and move on.
- **Notebook has no observable techniques (just plotting / EDA).** Mark `low_signal: true` and skip — but record any new external dataset references.

## Delegation to an external LLM

If the kernel count and notebook lengths exceed the agent's context budget for this stage, use external-tools handoff #2: bundle the relevant kernel folders, hand off to a long-context external model with the `ideas_pool.md` format spec, paste the returned entries back, then verify each cited `ref` exists in `kernels_index.json` before merging.

The agent itself **never** delegates code generation to an external tool from this stage. Distillation produces natural-language descriptions only.
