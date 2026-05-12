# External tool handoff

Several steps in the pipeline are better delegated to an external tool than executed inside the agent's main loop. This file lists every supported handoff point, what the external tool gets, what it must return, and which integrity rules still apply.

The orchestrator never invokes these tools silently. Each handoff produces a prompt file under `runs/<comp_slug>/handoff_prompts/`, and the orchestrator pauses (or proceeds without the handoff) according to `run.yaml.external_tools`.

---

## 1. Kaggle web scraping → web-fetch / browser tool

**When:** Stage 1 recon wants discussion threads and notebook *output* (not just the source), or the Kaggle API does not surface a field (e.g. submission-count UI text on the comp page).

**Inputs:** `https://www.kaggle.com/competitions/<slug>/discussion?sort=hotness`, or a specific notebook URL.

**Output:** Markdown summary saved to `stage1_recon/scraped/<slug>-<date>.md` with the original URL at the top.

**Integrity:** Scraped content is treated like a public kernel for citation purposes — every idea pulled from a discussion thread must appear in `citations.bib` with the discussion URL and post author.

**Tools that fit:** Claude Code's `WebFetch`, Playwright/Puppeteer, `curl + readability`. Pick whichever is available; the orchestrator does not care.

---

## 2. Top-kernel summarization → external strong LLM (optional)

**When:** Stage 1 recon has pulled 20+ notebooks and the orchestrator would have to read each to extract techniques. Delegating saves context budget.

**Inputs:** A folder of `.ipynb` files + a one-sentence task description ("classify which techniques each notebook uses, deduplicate across notebooks, return a unified ideas pool").

**Output:** `stage1_recon/ideas_pool.md` in the format defined in `state-contract.md`.

**Integrity:** The external model **must not** be given the comp's private data, or any user identifiers beyond the kaggle handles already public in the kernels. Re-implementation of techniques happens inside the agent, never the external tool — the external tool only extracts ideas, never writes code that lands in `pipeline.py`.

**Tools that fit:** Gemini 1.5 / 2.0 Pro (long context), Claude via API, GPT-4-class models. The orchestrator writes a self-contained prompt to `handoff_prompts/recon_summarize.md`; the user runs it through whichever tool and pastes the result back, or sets `run.yaml.external_tools.recon_summarize: auto` to invoke via API key.

---

## 3. Feature engineering brainstorm → external strong LLM (optional)

**When:** Stage 2 modeling is starting on a tabular problem and wants a brainstorm of candidate features beyond what the recon ideas already cover.

**Inputs:** Schema + first 50 rows of `train.csv` (sensitive PII redacted if present) + the target column name + the metric.

**Output:** Markdown list of 20–50 candidate features, each with: name, formula in pseudo-code, hypothesis for why it helps, expected computational cost.

**Integrity:** The agent re-implements every accepted feature inside `pipeline.py` (Rule 1). The brainstorm list goes into `stage2_modeling/feature_ideas.md` as a reference, not as code.

---

## 4. Vision augmentation policy → external vision agent (optional)

**When:** Stage 2 modeling is starting on a CV problem and wants a competition-tuned augmentation policy beyond standard `albumentations` defaults.

**Inputs:** Task type, dataset description, image samples (≤8 representative images), the metric, and a list of augmentations already tried (initially empty).

**Output:** An `albumentations.Compose([...])`-style policy spec in YAML, with rationale per augmentation.

**Integrity:** Translation from YAML to code happens in the agent. Policies are tracked in `stage2_modeling/augmentation_versions.yaml` so ablations are clean.

---

## 5. Hyperparameter search → Optuna subprocess

**When:** Stage 2 modeling wants more than a handful of HP trials and the search would dominate the agent's time.

**How:** The agent writes an Optuna study script `stage2_modeling/runs/<run_id>/optuna_study.py`, launches it as a subprocess (`python optuna_study.py --n-trials 50 --timeout 6h`), and polls the resulting `optuna.db` via `optuna.load_study()`. The agent itself can yield to the supervisor while the study runs.

**Integrity:** Trials and their seeds are logged in `progress.jsonl`. Final best params land in `runs/<run_id>/config.yaml` with `source: optuna_study`.

---

## 6. Notebook conversion for code-only submission → code-gen tool

**When:** The competition is code-only (`comp_profile.submission.code_only: true`) and the user's pipeline must be packaged as a single self-contained Kaggle notebook.

**Inputs:** The local pipeline (`stage2_modeling/pipeline.py` + any helper modules + the chosen run's `config.yaml`).

**Output:** A single `.ipynb` with no external dependencies beyond Kaggle Notebook's default environment.

**Integrity:** The conversion **must not** alter logic — only repackaging. The agent diffs the converted notebook's predictions against the local pipeline's on a held-out validation slice; mismatches block the submission.

**Tools that fit:** A strong code-gen model with the local pipeline as context, or `nbconvert` + manual stitching.

---

## 7. Pre-submission sanity check → independent reviewer

**When:** Right before any submission that is a candidate for one of the 2 final submissions.

**Inputs:** Summary of the submission's training config, CV scheme, public LB so far, and how it compares to top public kernels. Plus the integrity rules in `integrity-rules.md`.

**Output:** A go / no-go verdict with up to 3 specific concerns, written to `stage3_submit/sanity_<submission>.md`.

**Integrity:** This is advisory; the user still chooses. A `no-go` does not block submission, but it must appear in `recommendations.md` next to that candidate.

**Tools that fit:** Claude Code's sub-agent system, or a separately-invoked Claude/Gemini session.

---

## Handoff prompt template

Every handoff prompt file (`runs/<comp_slug>/handoff_prompts/<task>.md`) has the same shape:

```markdown
# Task
<one-paragraph description of what the external tool needs to do>

# Context
- Competition: <slug>
- Stage: <stageN_*>
- Integrity constraints: <pointers to rules in integrity-rules.md>

# Inputs
<inline content or file references>

# Expected output
<schema; path where the output should be written>

# Do NOT
- Do not access the user's Kaggle credentials.
- Do not access private data outside the listed inputs.
- Do not write code into pipeline.py — only into the listed output file.
```

The orchestrator prints `HANDOFF_READY <task> <path>` to stdout and proceeds (or waits, per `run.yaml.external_tools.<task>`).
