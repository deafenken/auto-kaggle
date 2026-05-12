---
name: auto-kaggle
description: >-
  Orchestrate a fully-autonomous, multi-day Kaggle medal-hunting pipeline from a
  competition URL to ranked submission candidates. At start it asks the user for
  the compute environment, downloads the data, parses the rules, and detects the
  task type. It then periodically scrapes top public kernels for ideas (with
  attribution), builds its own pipeline with CV-aware training, tracks the daily
  submission quota in UTC, and presents ranked submission recommendations for the
  user to pick from. Designed to survive interruptions: all state lives under
  runs/<comp_slug>/, every micro-step is logged, and the supervisor.sh script
  re-invokes the agent across crashes until deadline or a STOP file appears.
  Delegates to four sub-skills: auto-kaggle-bootstrap, auto-kaggle-recon,
  auto-kaggle-modeling, auto-kaggle-submit.
---

# Auto-Kaggle Orchestrator

A four-stage closed-loop Kaggle agent designed for **稳银冲金** (lock silver, aim for gold) targets. The orchestrator does **not** do the modeling itself — it sequences four specialist sub-skills, owns the state contract, and enforces the integrity rules.

Designed around three hard realities:

1. A Kaggle medal run takes **days to months**, not minutes. The agent process **will** be interrupted. Everything resumes from disk.
2. Public LB ≠ Private LB. Chasing public-LB tops blindly is how good runs become shake-up casualties. CV-first selection is non-negotiable.
3. Final 2 submissions are **user-picked**. The skill ranks; the user decides.

## When to invoke this skill

Trigger when the user says any of:

- "刷 Kaggle <competition URL or slug>"
- "auto kaggle <slug>"
- "帮我打 <kaggle comp>"
- "继续刷 <slug>" / "resume <slug>"
- Provides a Kaggle competition URL with a tier goal ("冲银" / "稳金" / "保铜")
- Asks "status of my <slug> run" — this triggers a status-only invocation (read `.heartbeat` + `recommendations.md`, print, exit; do not start any new work)

If the user provides only an idea or a finished pipeline (not a Kaggle URL), this is **not** the right skill — route to a generic modeling skill instead.

## High-level flow

```
                  ┌──────────────────────────────────────────────────────────┐
                  │       auto-kaggle  (this skill — orchestrator)           │
                  └──────────────────────────────────────────────────────────┘
                                       │
   First invocation? ─────yes──────►  Stage 0 (auto-kaggle-bootstrap)
                                       │   asks compute env, downloads data,
                                       │   parses rules, detects task type
                                       ▼
                       ┌───── periodic recon ─────►  Stage 1 (auto-kaggle-recon)
                       │                               pull top public kernels,
                       │                               distill ideas with citations
                       │                               (every N hours, configurable)
                       ▼
                  Stage 2 (auto-kaggle-modeling)
                     own pipeline + CV-aware training
                     + incorporate recon ideas with ablations
                       │
                       ▼
                  Stage 3 (auto-kaggle-submit)
                     rank candidates by trust-adjusted CV,
                     check quota, write recommendations.md,
                     wait for user pick, submit, log,
                     update quota_state.yaml
                       │
                  quota exhausted? ──yes──► write wait_until.txt, exit
                       │ no
                       └──► loop: more modeling / next recon / next submit
```

## Resume protocol (default behavior)

`auto-kaggle <comp_slug>` always means **resume**. Procedure:

1. Read `runs/<comp_slug>/run.yaml`. If missing → this is a first invocation, go to "First-invocation flow" below.
2. Check `STOP` / `PAUSE` → exit or idle if present.
3. Read `.heartbeat`. If `pid` is alive and `ts_utc` is fresh (<5 min), refuse to start a second agent — exit with message.
4. Read `stage3_submit/wait_until.txt`. If present and `now < wait_until`, print `STILL_WAITING …`, exit 0.
5. Read the last event per stage from `progress.jsonl` (sorted by `ts_utc`).
6. Dispatch:
   - Last event is `awaiting_user_pick` → print `recommendations.md`, exit waiting.
   - Last event is mid-stage → resume that stage from the next sub-step.
   - All stages have completed at least one cycle → run the loop body: maybe recon, train new ideas, write recommendations.
7. Update `.heartbeat` at start, every 60s during, and once at exit.

A fresh start requires explicit `--restart`. The orchestrator then prompts the user once more before deleting state.

## First-invocation flow

1. Ask the user (in this order):
   - Competition URL or slug (if not in the trigger).
   - **Compute environment** — show the 4 options from `auto-kaggle-bootstrap/references/compute-environment.md` and let the user pick.
   - Kaggle username (for attribution in submission messages).
   - Target tier (default: `silver-floor-gold-ceiling`).
   - Supervisor mode (default: `manual`; recommend `claude-loop` or `shell-supervisor` once they trust it).
2. `mkdir -p runs/<comp_slug>/`, write `run.yaml`.
3. Delegate to `auto-kaggle-bootstrap`.
4. After bootstrap returns, delegate to `auto-kaggle-recon` for the initial pull.
5. Delegate to `auto-kaggle-modeling` for a baseline run.
6. Delegate to `auto-kaggle-submit` to produce the first recommendation (no submission yet — user reviews).
7. Print a summary of the current state and where the next invocation will pick up.

## Status-only invocation

When the user asks for status (`"<slug> 现在怎么样了"`, `"status <slug>"`):

```bash
cat runs/<slug>/.heartbeat
tail -n 20 runs/<slug>/progress.jsonl
cat runs/<slug>/stage3_submit/recommendations.md
cat runs/<slug>/stage3_submit/quota_state.yaml
```

Print these (or summaries) and **exit without doing any work**. No training, no recon, no submission. The user is checking in, not authorizing action.

## Integrity gate (mandatory at every hand-off)

Before delegating to the next stage, the orchestrator verifies:

1. The previous stage's `hand_off.md` exists and conforms to the 3-paragraph spec (`state-contract.md`).
2. The structured files named in `hand_off.md` exist and parse.
3. No `STOP` or `PAUSE` sentinel is present.
4. No escalation block is open in the latest `recommendations.md`.
5. For Stage 3: `submission_log.jsonl` is in sync with `kaggle competitions submissions -c <slug>` (run the reconcile script before allowing a submit).

If any check fails → escalate per `escalation-policy.md`, do not proceed.

## State contract (the only inter-stage interface)

Every stage reads and writes files under `runs/<comp_slug>/`. Full schema in `references/state-contract.md`. Quick view:

```
runs/<comp_slug>/
├── run.yaml
├── .heartbeat
├── progress.jsonl
├── STOP | PAUSE         # sentinels
├── data/                # raw + processed (gitignored)
├── stage0_bootstrap/    # comp_profile.yaml, rules_summary.md, compute_env.yaml, hand_off.md
├── stage1_recon/        # kernels_index.json, ideas_pool.md, citations.bib, hand_off.md
├── stage2_modeling/     # pipeline.py, cv_split.yaml, runs/<run_id>/, leaderboard.csv, hand_off.md
└── stage3_submit/       # submission_log.jsonl, quota_state.yaml, recommendations.md,
                         # wait_until.txt, final_selection.md, hand_off.md
```

## When to load which reference

Default: load nothing extra. Load the files below only when making the decision they govern.

| File | Load when |
|---|---|
| `references/state-contract.md` | Setting up `runs/<comp_slug>/`, parsing an existing run, or writing any state file |
| `references/integrity-rules.md` | Before every stage hand-off (mandatory) and before every submission |
| `references/long-running-protocol.md` | On resume, on supervisor setup, when a stage skill plans to exit and yield |
| `references/external-tools.md` | When considering delegating recon summarization, feature brainstorm, augmentation policy, notebook conversion, or pre-submit review |
| `references/escalation-policy.md` | When something feels off — the question "should I escalate?" is answered here |
| `references/kaggle-cli-basics.md` | When wiring up a new `kaggle ...` command (especially auth or rate-limit issues) |

## Supervisor selection

The orchestrator does **not** start the supervisor itself. It tells the user how to start it, based on `run.yaml.supervisor.mode`:

- `manual` → "I will pause after each cycle. Invoke me again with `/auto-kaggle resume <slug>` when ready."
- `claude-loop` → "Inside Claude Code, run `/loop /auto-kaggle resume <slug>` to keep me cycling."
- `shell-supervisor` → "Run `nohup bash auto-kaggle/assets/supervisor.sh <slug> > supervisor.log 2>&1 &` to keep me running across crashes. See `supervisor.sh --help` for options."

The first invocation prints all three so the user can choose.

## Walkthrough — first 24h of a real run

1. User: `auto kaggle https://www.kaggle.com/competitions/playground-series-s4e5`
2. Orchestrator: prompts for compute env, username, tier. User picks `local-gpu` (1×3090), tier `silver-floor-gold-ceiling`.
3. Bootstrap: downloads ~400MB data, detects `tabular-regression`, metric `RMSE`, daily quota 5, deadline in 27 days.
4. Recon: pulls top 30 kernels by votes. Ideas pool lands with 14 deduplicated techniques (KFold=5, target log-transform, 3 specific feature engineerings, CatBoost+LGBM blend, etc.).
5. Modeling: builds a 5-fold LightGBM baseline. CV RMSE 0.745. Writes `runs/.../stage2_modeling/runs/2026-05-12-lgbm-baseline/`.
6. Submit: writes `recommendations.md` ranking the single baseline as the top candidate, prints quota `0/5 used`. **Does not submit** — first invocation always pauses for user.
7. User reviews `recommendations.md`, says "submit candidate 1." Orchestrator submits, logs to `submission_log.jsonl`, updates quota to `1/5`, prints public LB once available.
8. User leaves. Orchestrator continues to model new ideas. Hits 5/5 around 18:00 UTC. Writes `wait_until.txt: 2026-05-13T00:00:00Z`. Exits.
9. Next cycle (supervisor or user-triggered) at 00:00 UTC: resets quota, refreshes recommendations, prints "ready to submit, 0/5 used."

## Notes

- This skill is for medal-hunting on real, currently-open competitions. For practicing with finished comps (no leaderboard), use `--practice` to skip submission tracking.
- Knowledge competitions and tutorial comps (no medals offered) are explicitly out of scope — the skill will refuse them in bootstrap.
- The skill assumes you have **already accepted the competition rules on the Kaggle website**. If a 403 hits during data download, bootstrap escalates.
- The skill does not handle the very last hour before deadline autonomously. Six hours out it switches to "deadline mode" and gates every submission on user confirmation (see `integrity-rules.md` Rule 9).
