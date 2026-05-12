---
name: auto-kaggle-recon
description: >-
  Stage 1 of auto-kaggle. Periodically pulls the highest-voted and highest-
  public-LB public kernels for the target competition, downloads their source
  for offline reading, and distills the techniques used into an attributed
  ideas_pool.md (deduplicated across kernels). Never copies code verbatim —
  re-implementation happens in Stage 2 modeling. Maintains citations.bib so
  every Stage 3 submission can name its sources. Throttled by last_recon_at
  to respect Kaggle CLI rate limits. Idempotent: re-running with no new
  kernels is a no-op. Designed to survive interruptions — pull state is
  recoverable from kernels_index.json + last_recon_at.
---

# Stage 1 — Recon

Pulls public kernels for the competition on a schedule, then distills them into a structured "ideas pool" with attribution. The pool is the menu Stage 2 picks from.

This stage **never produces submission code**. It only produces *ideas about what to try*, each labeled with the kernel(s) where it appeared. Stage 2 is responsible for re-implementing those ideas from scratch.

## Trigger

- Delegated to by `auto-kaggle` after bootstrap completes (initial pull), then again every `run.yaml.recon_interval_hours` (default 6).
- Direct: `auto-kaggle-recon <comp_slug>` to force a pull (still throttled — pass `--force` to override).
- Status: `auto-kaggle-recon status <comp_slug>` prints the index summary and exits.

If `now - last_recon_at < interval_hours` and no `--force`, exit no-op with a one-line message.

## Inputs

- `runs/<comp_slug>/stage0_bootstrap/comp_profile.yaml`
- `runs/<comp_slug>/stage0_bootstrap/hand_off.md`
- `runs/<comp_slug>/stage1_recon/last_recon_at` (if exists)
- `runs/<comp_slug>/stage1_recon/kernels_index.json` (if exists — used for delta diffing)

## Outputs (contract — full schemas in `auto-kaggle/references/state-contract.md`)

```
runs/<comp_slug>/stage1_recon/
├── kernels_index.json          # full index of pulled kernels (votes + public_lb + last_updated)
├── kernels/<author>__<slug>/   # one folder per pulled kernel (source + metadata)
├── ideas_pool.md               # deduplicated techniques with per-kernel attribution
├── citations.bib               # BibTeX-style entries for every referenced kernel
├── external_data_candidates.md # external datasets the kernels rely on (user reviews)
├── last_recon_at               # ISO-8601 UTC of the most recent pull
└── hand_off.md                 # briefing for Stage 2
```

## Workflow

### Step 0 — throttle check

Read `last_recon_at`. If it exists and `now - last_recon_at < run.yaml.recon_interval_hours` and the caller did not pass `--force`:

- Print `recon throttled — last pull was Nh ago, next allowed in Mh` and exit 0.
- Do not touch any output file.

### Step 1 — two listings

Pull two CSV listings via the helper:

```bash
kaggle kernels list -c <slug> --sort-by voteCount --page-size 50 --csv > .recon_tmp/by_votes.csv
kaggle kernels list -c <slug> --sort-by scoreDescending --page-size 50 --csv > .recon_tmp/by_score.csv
```

Both go through the rate-limited wrapper in `kaggle_helpers.py`. Space subsequent calls by ~1s.

Append `progress.jsonl` event `recon_listing_pulled` for each listing with the row count.

### Step 2 — union, dedupe, diff against existing index

The two listings overlap heavily; deduplicate by `ref` (Kaggle's `author/kernel-slug`). For each kernel, record:

```json
{
  "ref": "jdoe123/eda-and-lgbm-baseline",
  "author": "jdoe123",
  "slug": "eda-and-lgbm-baseline",
  "title": "EDA and LGBM Baseline",
  "votes": 142,
  "public_lb": 0.7488,
  "last_run_time_utc": "2026-05-10T15:23:00Z",
  "last_pulled_utc": null,
  "in_top_votes": true,
  "in_top_score": false
}
```

Load the previous `kernels_index.json`. For each kernel in the new union, classify:

| Status | When |
|---|---|
| `new` | not in previous index |
| `updated` | in previous index AND `last_run_time_utc` is newer than `last_pulled_utc` |
| `unchanged` | in previous index AND not updated |
| `dropped` | in previous index but not in either new listing |

`new` + `updated` kernels are pulled in Step 3. `dropped` kernels stay in the index (marked `dropped_at_utc`) — Stage 3 may still cite them.

### Step 3 — pull deltas

For each `new` or `updated` kernel:

```bash
kaggle kernels pull <ref> -p runs/<slug>/stage1_recon/kernels/<author>__<slug>/ -m
```

The `-m` flag pulls metadata too (license, kernel type, tags). Set `last_pulled_utc` to now.

Cap each recon cycle at 20 new pulls. If more remain, prioritize by `votes + 100 * has_public_lb`, drop the rest until the next cycle. Log `progress.jsonl` event `recon_pull_capped` if this triggers.

Append `progress.jsonl` event `kernel_pulled` per pulled kernel.

### Step 4 — distillation (agent does this, not the script)

This is the step the script **cannot** do — extracting techniques from a notebook needs reading comprehension. The agent reads each pulled kernel's source from `kernels/<author>__<slug>/` and adds one or more entries to `ideas_pool.md`.

The exact entry format is in `references/ideas-pool-format.md`. Each entry has:

- A short imperative name ("StratifiedKFold with N=5, shuffle=True, seed=42")
- Category tag (`cv` / `feature` / `model` / `aug` / `ensemble` / `post` / `external_data`)
- One-paragraph description of what and why
- Quantitative effect if the kernel reports one ("+0.003 CV in their ablation")
- Attribution: 1 or more `ref` strings from `kernels_index.json`
- Estimated implementation cost (`S` / `M` / `L`)
- Cross-references: ideas it composes with, ideas it conflicts with

Dedupe: if the same technique appears in N kernels, it becomes ONE entry with N attributions. A technique mentioned by ≥3 top-10 kernels is flagged `consensus: true`.

When the agent finishes a kernel's distillation, it appends a comment line in `kernels_index.json` for that kernel: `"distilled_at_utc": "..."`.

Delegate to an external strong LLM if the kernel count and notebook length exceed the agent's context budget — see `auto-kaggle/references/external-tools.md` handoff #2.

### Step 5 — citations and external data

For every distilled kernel, append a BibTeX-style entry to `citations.bib`:

```bibtex
@misc{jdoe123-eda-and-lgbm-baseline,
  author = {jdoe123},
  title  = {EDA and LGBM Baseline},
  year   = {2026},
  url    = {https://www.kaggle.com/code/jdoe123/eda-and-lgbm-baseline},
  note   = {Kaggle kernel, pulled 2026-05-12, public LB 0.7488, votes 142, license Apache-2.0},
  key    = {jdoe123-eda-and-lgbm-baseline}
}
```

Stage 3 submission messages use the `key` as their `attr:` value.

For each external dataset referenced by a kernel (look at `metadata.json` from `kernels pull` — `dataSources` field), add an entry to `external_data_candidates.md`:

```markdown
## openimagesv7-validation-subset

- Source kernel(s): jdoe123/eda-and-lgbm-baseline, msmith/heavier-baseline
- Kaggle dataset ID: openimages/openimagesv7-validation-subset
- Size: 2.4 GB
- License: CC BY-4.0
- Why useful: provides held-out labels missing from competition train set
- **Approved for use:** NO  ← user flips this to YES after reading comp rules
```

Integrity rule 10: Stage 2 may use only datasets with `Approved for use: YES`. This stage **never** flips that flag — the user does.

### Step 6 — finalize

1. Atomic-rewrite `kernels_index.json`.
2. Atomic-rewrite `last_recon_at` with the current ISO-8601 UTC timestamp.
3. Append `progress.jsonl` event `recon_pulled` with `{kernels_total, new, updated, dropped, ideas_total}`.
4. Update `hand_off.md` per the contract:

```markdown
# Stage 1 → Stage 2 hand-off (recon cycle <N>, pulled <ts>)

## What I did
- Pulled <N> new kernels, refreshed <M> updated kernels, dropped <K> stale entries.
- Distilled <I> ideas (now <T> total in pool, <C> with consensus across ≥3 kernels).
- Found <X> external dataset candidates, of which <Y> are pre-approved.

## What's true now
- Top public LB among pulled kernels: <score> (held by <ref>).
- Ideas with consensus: <bullet list of top 5 by # citations>.
- Ideas new this cycle: <bullet list of all new>.
- External data: <list of newly-flagged candidates needing user review>.

## What you should do next
Stage 2: prioritize the consensus ideas not yet covered in any
attribution.md under stage2_modeling/runs/*/. Then the highest-cost-effective
new ideas. Estimate wallclock before starting any run.
```

5. Update `.heartbeat` and exit.

## Idempotency

- Running with no new kernels (`new=0, updated=0`) is a no-op for everything except updating `last_recon_at` and appending a single `recon_pulled` event.
- A pull interrupted mid-kernel leaves `.tmp` files in `kernels/<author>__<slug>/`. On resume, those files are deleted and the kernel is re-pulled.
- The distillation step is per-kernel idempotent via the `distilled_at_utc` field — already-distilled kernels are skipped unless `--re-distill` is passed.

## Failure modes

| Symptom | Action |
|---|---|
| `kaggle kernels list` returns 429 | Helper backs off (60s / 5min / 30min), retries. If all 3 fail, exit and let supervisor retry next cycle. |
| A specific kernel fails to pull (404, deleted by author) | Mark in index as `pull_failed: true` with timestamp, skip, continue with rest. |
| Distillation returns no ideas for a kernel | Mark kernel as `low_signal: true`. Do not retry next cycle. |
| `kernels_index.json` cannot be parsed on resume | Move it to `kernels_index.json.broken.<ts>` and start fresh. Log `progress.jsonl` event `index_recovered`. |

## When to load which reference

| File | Load when |
|---|---|
| `references/kernel-distillation.md` | Step 4 — extracting techniques from a notebook |
| `references/ideas-pool-format.md` | Step 4 — writing the entry, especially category tags + deduplication |
| `references/throttle-and-rate.md` | Step 0 / step 1 — throttle window, rate limit handling |
| `auto-kaggle/references/state-contract.md` | Always |
| `auto-kaggle/references/integrity-rules.md` | Step 5 — citations + external data (Rule 1, Rule 10) |
| `auto-kaggle/references/kaggle-cli-basics.md` | Every `kaggle ...` call |
| `auto-kaggle/references/external-tools.md` | Delegating distillation to an external LLM (handoff #2) |
| `auto-kaggle/references/long-running-protocol.md` | On resume / mid-pull recovery |
