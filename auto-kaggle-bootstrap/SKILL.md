---
name: auto-kaggle-bootstrap
description: >-
  Stage 0 of auto-kaggle. Asks the user which compute environment to use
  (Kaggle Notebook free / local GPU / cloud GPU / CPU-only), downloads the
  competition data via the Kaggle CLI, parses the Overview / Rules /
  Evaluation pages to extract task type, metric, submission format, daily
  quota, and deadline, then writes comp_profile.yaml and hand_off.md so
  later stages have a single source of truth for what the competition is.
  Refuses to bootstrap knowledge-only / tutorial competitions that offer
  no medals.
---

# Stage 0 — Bootstrap

The first thing that happens after `auto-kaggle <slug>`. Turns "user gave me a URL" into "every later stage knows what kind of competition this is and what we're allowed to do."

## Trigger

- Delegated to by the `auto-kaggle` orchestrator on first invocation.
- Can be invoked directly: `auto-kaggle-bootstrap <comp_slug>` to re-derive the comp profile (e.g. after Kaggle updated the rules mid-comp).

Never invoke this stage on a run that has already passed bootstrap unless the user explicitly asked to re-bootstrap.

## Inputs

- `runs/<comp_slug>/run.yaml` — created by the orchestrator with `comp_slug`, `url`, `kaggle_username`, `target_tier`, `supervisor`.
- The user (interactively) — only for the 4 questions in step 1 below. Never for technical decisions the agent can make itself.

## Outputs (the contract)

```
runs/<comp_slug>/
├── data/
│   └── raw/                       # output of kaggle competitions download
├── stage0_bootstrap/
│   ├── comp_profile.yaml          # the structured profile every stage reads
│   ├── rules_summary.md           # human-readable distillation of Overview + Rules + Evaluation
│   ├── data_stats.md              # row counts, file sizes, column types, target distribution
│   ├── compute_env.yaml           # user's chosen env + capability vector
│   ├── raw_comp_view.txt          # cached output of `kaggle competitions view <slug>`
│   └── hand_off.md
```

Exact schemas: `auto-kaggle/references/state-contract.md`.

### Two-phase write

`bootstrap.py` produces a **partial** `comp_profile.yaml` (with
`bootstrap_partial: true`) and a partial `hand_off.md` that names the gaps —
it cannot WebFetch the Rules / Evaluation pages itself. The agent then
completes Steps 1, 7, and 8 below: writing `compute_env.yaml` from the user's
answers, writing `rules_summary.md` from the fetched pages, and filling the
remaining `comp_profile.yaml` fields (metric direction, daily quota, team
rules, external data rules, code-only constraints). The agent flips
`bootstrap_partial: false` once everything is in.

The orchestrator's idempotency check (`bootstrap.py is_bootstrap_done`) only
returns true when **all of**: `comp_profile.yaml` exists with
`bootstrap_partial: false`, `rules_summary.md` exists, and `compute_env.yaml`
exists. A run with only the script's partial outputs will re-trigger the agent
finalization on the next invocation, not be silently treated as complete.

## Workflow

### 1. Ask the user 4 questions, in this order

Print exactly:

```
About to bootstrap <comp_slug>. I need 4 things before I start:

(a) Compute environment — pick one:
    1. kaggle-notebook   (free T4×2 or P100, 9h kernel limit, 30h GPU/week)
    2. local-gpu         (your machine; tell me GPU model + VRAM)
    3. cloud-gpu         (Colab/Lambda/Vast/RunPod; you manage the instance)
    4. cpu-only          (only viable for small tabular comps)

(b) Kaggle username (for attribution in submission messages).

(c) Target tier:
    - silver
    - silver-floor-gold-ceiling   (default; lock silver, push for gold)
    - gold

(d) Supervisor mode:
    - manual           (you invoke me each cycle)
    - claude-loop      (use /loop inside Claude Code)
    - shell-supervisor (assets/supervisor.sh runs outside Claude)
```

Wait for the user's answers. Validate (compute env must be one of the 4; tier from the list; supervisor from the list). If anything is ambiguous, ask once more; do not guess.

Write the answers into `run.yaml` (already exists, edit in place) and into `stage0_bootstrap/compute_env.yaml`:

```yaml
env: local-gpu                       # one of the 4
specs:
  gpu_model: RTX 3090
  vram_gb: 24
  cpu_cores: 16
  ram_gb: 64
  internet: true                     # always true except inside a code-only submission notebook
constraints:
  max_wallclock_per_run_hours: 10    # auto-derived: kaggle-notebook=9, local/cloud=user-supplied, cpu-only=4
  parallel_runs: 1
```

Compute env defaults if the user gives partial info: see `references/compute-environment.md` for the lookup table.

### 2. Confirm the comp exists and pull metadata

```bash
kaggle competitions view <slug> > stage0_bootstrap/raw_comp_view.json
```

If this returns non-zero, escalate (`escalation-policy.md` "Auth or comp does not exist"). Do not proceed.

Parse the output for: `title`, `deadline`, `category`, `reward`, `evaluationMetric`, `submissions/team`, `userHasEntered`.

### 3. Refuse non-medal competitions

If `category` is one of:
- "Getting Started" / "Playground" with no medal tier
- "Knowledge"
- "Tutorial"

→ Write a one-paragraph hand_off.md explaining the comp does not award medals and exit. The orchestrator on receiving this will inform the user and stop.

Exception: `Playground Series` competitions on the modern Kaggle Playground track **do** award medals — verify by checking the comp page for the medal-tier text.

### 4. Detect task type, metric, submission format

The detection sources, in priority order:

1. `kaggle competitions view`'s `evaluationMetric` field.
2. The contents of `sample_submission.csv` (column names) and `test.csv` (if present) inside `data/raw/`.
3. The comp's Overview page (fetched via WebFetch if needed).

The full detection rule-set lives in `references/task-type-detection.md`. The result is one of the values listed in `state-contract.md` under `comp_profile.task_type`.

Special cases:
- If the metric is custom (e.g. "competition-specific F-beta with weights"), record the description verbatim in `comp_profile.metric.description` and look for an evaluation script in the dataset — code-only comps often ship one.
- If the submission format is "kernel-only" (no CSV upload allowed), set `comp_profile.submission.code_only: true`. This propagates a constraint forward — Stage 2 modeling will eventually have to produce a self-contained notebook.

### 5. Download the data

```bash
mkdir -p runs/<comp_slug>/data/raw/
kaggle competitions download -c <slug> -p runs/<comp_slug>/data/raw/
cd runs/<comp_slug>/data/raw/ && unzip -o "*.zip" && rm -f *.zip
```

Log to `progress.jsonl` event `data_downloaded` with the total bytes.

If the user has not accepted the competition rules on the Kaggle website, this command returns 403. Escalate per `kaggle-cli-basics.md` — do not retry.

### 6. Quick data stats

For every CSV / Parquet under `data/raw/`:
- row count
- column count + dtypes
- missingness per column (% null)
- if a target column is identified, target distribution summary

For image / NLP / time-series data:
- file count, total size, top-level folder structure
- sample 5 random items and record their shape / token count / time range

Write all of this to `stage0_bootstrap/data_stats.md`. Keep it under 100 lines; this file is read by every later stage.

### 7. Rules summary

Fetch the Overview / Rules / Evaluation pages (via WebFetch — they are not all in the API output). Extract:

- Daily submission limit (almost always 5, but some comps differ).
- Team size limit.
- Solo-only flag.
- External data rules (allowed? must be Kaggle-shared?).
- Code-only constraints (runtime cap, internet allowed, max submission file size).
- Deadline (UTC).
- Special anti-cheating clauses (e.g. "no LB probing", "no probing test set indices").
- Prize / medal tiers.

Write `rules_summary.md` with these as bullet points + the exact quoted text for anything ambiguous.

### 8. Write `comp_profile.yaml` and `hand_off.md`

Both files follow `state-contract.md` schemas. The hand_off briefs Stage 1:

```markdown
# Stage 0 → Stage 1 hand-off

## What I did
- Wrote comp_profile.yaml, rules_summary.md, data_stats.md, compute_env.yaml.
- Downloaded <bytes> of data to runs/<slug>/data/raw/.
- Detected: task_type=<...>, metric=<...> (direction=<min|max>), daily_quota=<N>, deadline=<UTC>.

## What's true now
- Compute env is <env>. Wallclock cap per run: <hours>h.
- Submission format: <csv|code-only>. Columns: <list>.
- External data: <allowed|forbidden>. Must be Kaggle-shared: <yes|no>.
- Today's UTC midnight: <ts>. Days until deadline: <N>.

## What you should do next
Stage 1: pull the top 30 public kernels by votes and the top 30 by public LB.
De-duplicate by kernel ID. For each, extract techniques used and add to
ideas_pool.md with citations. Aim for 15–25 distinct ideas after dedup.
Initial recon, so set last_recon_at to now and do not re-pull for at least
6 hours unless told otherwise.
```

Append `progress.jsonl` event `bootstrap_done`. Update `.heartbeat` with `stage: done`. Exit.

## Idempotency

If `comp_profile.yaml` already exists when this skill runs:

- Compare its `deadline_utc` to `kaggle competitions view`'s current deadline. If different → competition was rescheduled, regenerate.
- Compare `quota.daily_limit` to current rules. If different → re-fetch and overwrite.
- Otherwise: log `progress.jsonl` event `bootstrap_skipped`, print "already bootstrapped, nothing to do" and exit 0.

A user can force regeneration with the orchestrator's `--re-bootstrap` flag.

## Failure modes

| Symptom | Action |
|---|---|
| `kaggle competitions view` returns 401/403 | Escalate per `kaggle-cli-basics.md`. Do not retry. |
| `kaggle competitions download` returns 403 | Most likely: user has not accepted comp rules. Escalate with the exact message. |
| Detection finds zero plausible task types | Escalate: dump the comp's sample submission + first 5 rows of test, ask the user. |
| Data exceeds compute_env's disk / RAM by 2× | Escalate. Suggest changing compute env. Do not download more than 80% of free disk. |
| Comp is team-only and user has not specified team | Escalate. Ask for `run.yaml.team` list. |

## When to load which reference

| File | Load when |
|---|---|
| `references/compute-environment.md` | Step 1 — building the option list and parsing the user's answer |
| `references/task-type-detection.md` | Step 4 — disambiguating the metric / submission format |
| `references/rules-parsing.md` | Step 7 — extracting the bits from the Overview / Rules pages |
| `auto-kaggle/references/state-contract.md` | When writing any output file |
| `auto-kaggle/references/integrity-rules.md` | Always — Rule 10 (external data) applies during bootstrap |
| `auto-kaggle/references/kaggle-cli-basics.md` | Every `kaggle ...` invocation |
