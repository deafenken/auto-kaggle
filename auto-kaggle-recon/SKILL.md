---
name: auto-kaggle-recon
description: >-
  Stage 1 of auto-kaggle (NOT YET IMPLEMENTED — design only). Periodically
  scrapes the highest-voted and highest-public-LB public kernels for the
  target competition, pulls their source for offline reading, and distills
  the techniques used into an attributed ideas_pool.md. Never copies code
  verbatim — re-implementation happens in Stage 2 modeling. Maintains
  citations.bib so every submission can name its sources. Throttled by
  last_recon_at to respect the Kaggle CLI rate limits.
---

# Stage 1 — Recon (DESIGN PLACEHOLDER)

> **Status:** contract only — implementation lands in the next delivery. The orchestrator and Stage 0 already write the inputs this stage will read. This file fixes the contract so Stage 0 and Stage 2 do not drift.

## Trigger

- Delegated to by `auto-kaggle` after bootstrap completes, then re-invoked periodically (every `run.yaml.recon_interval_hours`, default 6).
- Can be invoked directly: `auto-kaggle-recon <comp_slug>`.

If `last_recon_at` is younger than the throttle window, exit no-op.

## Inputs

- `runs/<comp_slug>/stage0_bootstrap/comp_profile.yaml`
- `runs/<comp_slug>/stage0_bootstrap/hand_off.md`
- `runs/<comp_slug>/stage1_recon/last_recon_at` (if exists)
- `runs/<comp_slug>/stage1_recon/kernels_index.json` (if exists — used for delta)

## Outputs (contract — exact schemas in `auto-kaggle/references/state-contract.md`)

```
runs/<comp_slug>/stage1_recon/
├── kernels_index.json    # top public kernels: id, author, votes, public_lb, last_updated
├── kernels/<author>__<slug>/   # downloaded notebook + metadata for each top kernel
├── ideas_pool.md         # deduplicated techniques with per-kernel attribution
├── citations.bib         # BibTeX-style entries for every referenced kernel
├── last_recon_at         # ISO-8601 UTC timestamp of last pull
└── hand_off.md           # briefing for Stage 2
```

## Workflow (planned)

1. Throttle check — exit no-op if `now - last_recon_at < interval`.
2. Pull two listings: `--sort-by voteCount` (top 30) and `--sort-by scoreDescending` (top 30 by public LB).
3. Diff against `kernels_index.json` from the previous pull; pull only new or updated kernels.
4. For each new/updated kernel: `kaggle kernels pull` into `kernels/<author>__<slug>/`.
5. For each kernel: extract techniques (KFold scheme, model family, augmentations, post-processing, external data used) into `ideas_pool.md` with `attribution: <author>/<slug>` per idea.
6. Deduplicate ideas across kernels — same technique mentioned by N kernels gets one entry with all N citations.
7. Update `citations.bib` and `last_recon_at`.
8. Write `hand_off.md` briefing Stage 2 with: top 3 new ideas, kernels that newly entered top-10, any kernels that disappeared from the pool.

## Integrity gates this stage enforces

- Rule 1 — extract ideas in natural language, not code. The `kernels/` folder is read-only reference; no file under it is ever copied into Stage 2.
- Rule 10 — every external dataset referenced by a kernel goes into `external_data_candidates.md` with its Kaggle dataset ID, so the user can decide whether to use it.

## When to load which reference (planned)

| File | Load when |
|---|---|
| `auto-kaggle/references/state-contract.md` | Always |
| `auto-kaggle/references/integrity-rules.md` | Before writing ideas_pool or citations |
| `auto-kaggle/references/kaggle-cli-basics.md` | Every `kaggle kernels ...` invocation |
| `auto-kaggle/references/external-tools.md` | When delegating kernel-summarization to an external LLM (handoff #2) |
| `auto-kaggle/references/long-running-protocol.md` | Resume / throttle logic |
