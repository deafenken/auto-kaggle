---
name: auto-kaggle-submit
description: >-
  Stage 3 of auto-kaggle (NOT YET IMPLEMENTED — design only). Ranks the
  submission candidates produced by Stage 2 by trust-adjusted CV (never
  raw public LB), shows the daily quota status with the next UTC reset
  time, writes recommendations.md, and waits for user pick. Runs the
  actual `kaggle competitions submit` with mandatory attribution in the
  message, appends to submission_log.jsonl, reconciles quota_state.yaml
  against `kaggle competitions submissions`. When daily quota is exhausted,
  writes wait_until.txt and exits so the supervisor sleeps. Final 2
  submissions before deadline are always user-gated.
---

# Stage 3 — Submit (DESIGN PLACEHOLDER)

> **Status:** contract only — implementation lands in the next delivery.

## Trigger

- Delegated to by `auto-kaggle` after a modeling run lands a new candidate.
- Can be invoked directly: `auto-kaggle-submit <comp_slug>` to print current recommendations / quota state without submitting.

## Inputs

- `runs/<comp_slug>/stage0_bootstrap/comp_profile.yaml`
- `runs/<comp_slug>/stage2_modeling/leaderboard.csv`
- `runs/<comp_slug>/stage2_modeling/runs/<run_id>/test_preds.csv` (for the candidates being considered)
- `runs/<comp_slug>/stage3_submit/submission_log.jsonl` (if exists)
- `runs/<comp_slug>/stage3_submit/quota_state.yaml` (if exists; otherwise reconciled from Kaggle)

## Outputs (contract)

```
runs/<comp_slug>/stage3_submit/
├── submission_log.jsonl    # append-only: every actual submission with kaggle_submission_id + public_lb
├── quota_state.yaml        # used_today, daily_limit, next_reset_utc, remaining, exhausted
├── wait_until.txt          # ISO-8601 UTC of next reset, when quota is exhausted
├── recommendations.md      # ranked candidates with reasoning, refreshed each cycle
├── final_selection.md      # the 2 final submissions the user picked, near deadline
└── hand_off.md
```

## Workflow (planned)

1. **Reconcile quota.** Run `kaggle competitions submissions -c <slug>`, count today's submissions in UTC, update `quota_state.yaml`. Kaggle is authoritative; local cache is hint only.
2. **Check wait state.** If `quota_state.exhausted: true` and `now < next_reset_utc` → write/refresh `wait_until.txt` and exit with `WAIT_UNTIL ...` line.
3. **Refresh recommendations.** Read `leaderboard.csv`, rank candidates by:
   - Trust-adjusted CV = `cv_score - α × cv_variance` (α from `comp_profile`)
   - Penalty for near-duplicates of recently submitted preds (Rule 4 enforcement)
   - Diversity bonus when picking 2nd / 3rd candidate
4. **Write `recommendations.md`** with top 5 candidates, each annotated with: CV, public LB if known, gap, attributed kernels, what's novel, risk notes. Top 1 is marked `RECOMMENDED`; 2nd is marked `SAFE` (lowest variance); user can pick anything in the list.
5. **Wait for user pick.** Append `progress.jsonl` event `awaiting_user_pick` and exit. Next invocation that finds the user has named a candidate (via a file write, the orchestrator's CLI arg, or a follow-up message) proceeds to step 6.
6. **Build attribution-checked submission message.** Format: `<short description> | attr: <author1>/<kernel1>, <author2>/<kernel2>`. Refuse to submit if message has no `attr:` token (Rule 2).
7. **Submit.** Call `kaggle competitions submit` via helper. Capture submission ID. Append to `submission_log.jsonl`.
8. **Post-submit.** Poll `kaggle competitions submissions` until public_lb appears (with backoff). Update the log entry. Update `quota_state.yaml`. Append `progress.jsonl` event `submitted` then `quota_used`.
9. **Quota exhausted?** If `used_today == daily_limit`: write `wait_until.txt`, append `progress.jsonl` event `quota_exhausted`, print `WAIT_UNTIL ...`, exit.
10. **Deadline mode.** If `now > deadline - 24h`: switch to deadline mode — refresh recommendations into `final_selection.md` instead, force user pick for **both** final submissions, never auto-pick.

## The "实时显示剩余" output

Every invocation (whether it submits or not) prints to stdout something like:

```
[auto-kaggle-submit] <slug>
  Quota: 3/5 used today  ·  Remaining: 2  ·  Next reset in 2h 17m (2026-05-13T00:00:00Z)
  Top candidate: 2026-05-12-lgbm-blend-seed3   CV 0.7421 (var 0.004)  public_lb (last) 0.7488
  Recommendations file: runs/<slug>/stage3_submit/recommendations.md
```

This single line is also the supervisor's progress signal — `supervisor.log` will accumulate one per cycle.

## Integrity gates this stage enforces

- Rule 2 — message attribution check before every `kaggle competitions submit`.
- Rule 3 — recommendations ranked by trust-adjusted CV, never by raw public LB.
- Rule 4 — submission blocked if predictions differ from a prior submission by < 1e-6 MAE; user must override with `--allow-near-duplicate` + written justification.
- Rule 6 — submission blocked if candidate's CV is worse than the median-prediction baseline.
- Rule 9 — in the final 6h: no auto-submission, every submit is user-gated; the 2 final submissions also user-gated.

## When to load which reference (planned)

| File | Load when |
|---|---|
| `auto-kaggle/references/state-contract.md` | Always |
| `auto-kaggle/references/integrity-rules.md` | Always — Rules 2/3/4/6/9 apply here |
| `auto-kaggle/references/kaggle-cli-basics.md` | Every `kaggle competitions submit` / `submissions` invocation |
| `auto-kaggle/references/long-running-protocol.md` | Resume / wait_until / heartbeat / append-only log |
| `auto-kaggle/references/escalation-policy.md` | Quota desync, near-duplicate, final-selection, deadline-mode |
