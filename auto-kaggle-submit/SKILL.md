---
name: auto-kaggle-submit
description: >-
  Stage 3 of auto-kaggle. Reconciles daily submission quota against
  `kaggle competitions submissions` (authoritative), ranks candidates from
  Stage 2 by trust-adjusted CV (never raw public LB), shows real-time
  "quota X/N remaining, next UTC reset in HH:MM", writes recommendations.md,
  and waits for user pick. Refuses to submit without attribution in the
  message (rule 2) or near-duplicate of prior submissions (rule 4). When
  daily quota hits zero, writes wait_until.txt with the next UTC midnight
  and exits so the supervisor sleeps. The final 2 submissions before
  deadline are always user-picked, never auto-submitted (rules 3 and 9).
---

# Stage 3 — Submit

The only stage that talks to Kaggle for submission. Other stages may pull metadata; this one is where bytes leave to the leaderboard.

## Trigger

- Delegated to by `auto-kaggle` whenever a Stage 2 run completes (or whenever there's a chance to refresh recommendations).
- Direct (refresh recs + show status): `auto-kaggle-submit <comp_slug>`.
- Direct (submit a user-chosen run): `auto-kaggle-submit <comp_slug> --submit <run_id>`.
- Direct (quota only, no submit): `auto-kaggle-submit <comp_slug> --quota-only`.

## Inputs

- `runs/<comp_slug>/stage0_bootstrap/comp_profile.yaml`
- `runs/<comp_slug>/stage2_modeling/leaderboard.csv`
- `runs/<comp_slug>/stage2_modeling/runs/<run_id>/test_preds.csv` (per candidate)
- `runs/<comp_slug>/stage2_modeling/runs/<run_id>/attribution.md` (per candidate)
- `runs/<comp_slug>/stage3_submit/submission_log.jsonl` (if exists; created by us)
- `runs/<comp_slug>/stage3_submit/quota_state.yaml` (if exists)

## Outputs (contract — full schemas in `auto-kaggle/references/state-contract.md`)

```
runs/<comp_slug>/stage3_submit/
├── submission_log.jsonl       # append-only: every actual Kaggle submission
├── quota_state.yaml           # used_today, daily_limit, remaining, next_reset_utc, exhausted
├── wait_until.txt             # next UTC midnight, when quota is exhausted
├── recommendations.md         # ranked candidates with reasoning, refreshed each cycle
├── final_selection.md         # the 2 final submissions the user picked at deadline
└── hand_off.md
```

## Workflow

### Step 1 — Reconcile quota from Kaggle

```bash
kaggle competitions submissions -c <slug> -v > .submit_tmp/submissions.txt
```

Parse the output, count rows whose date == today UTC. This is the authoritative count of today's submissions. Compare to `quota_state.yaml.used_today`. If they disagree by more than 1:

- Likely cause: the agent missed logging a submission (network blip during local log write), OR a teammate submitted from the same account, OR Kaggle counted a build-only failure as a submission.
- Action: overwrite `used_today` with Kaggle's count, **escalate** per rule (state corruption escalation in `escalation-policy.md`).

Otherwise update `quota_state.yaml`:

```yaml
used_today: 3
daily_limit: 5
remaining: 2
last_reset_utc: 2026-05-12T00:00:00Z
next_reset_utc: 2026-05-13T00:00:00Z
exhausted: false
```

Always print the one-line status:

```
[auto-kaggle-submit] <slug>  Quota 3/5  Remaining 2  Next reset in 2h 17m (2026-05-13T00:00Z)
```

### Step 2 — Check exhausted

If `exhausted: true` and `now < next_reset_utc`:

1. Write/refresh `wait_until.txt` containing the ISO-8601 UTC of `next_reset_utc`.
2. Append `progress.jsonl` event `quota_exhausted`.
3. Print `WAIT_UNTIL <ts> (in HH:MM)`.
4. Exit 0. The supervisor (mode `shell-supervisor` or `claude-loop`) sleeps until then.

If `now >= next_reset_utc`:

1. Reset `used_today = 0`, recompute `next_reset_utc`.
2. Delete `wait_until.txt`.
3. Append `progress.jsonl` event `quota_reset`.
4. Continue.

### Step 3 — Refresh `recommendations.md`

Read `leaderboard.csv`. For each `completed` run, compute `trust_adjusted_cv`:

```
trust_adjusted = cv_score   - α * cv_std   (if metric is maximized)
trust_adjusted = cv_score   + α * cv_std   (if metric is minimized)
```

Where `α` = 1.0 by default (configurable in `run.yaml.trust_alpha`). Higher α punishes high-variance runs more.

Additional candidate filters:

- **Near-duplicate penalty** (Rule 4): if a candidate's `test_preds.csv` has MAE < 1e-6 from any submission in `submission_log.jsonl`, mark `near_duplicate: true` and exclude from `RECOMMENDED` / `SAFE` slots (but keep in the ranked list with a note so the user can override).
- **Untrustworthy filter**: if `cv_std / |cv_score|` > 0.5, mark `untrustworthy: true`. Allowed in the list but not in the top slots.
- **Cross-validation status**: only `status == completed` rows are considered. `terminated_overbudget` / `failed` are filtered out.

Sort by `trust_adjusted` in the metric's preferred direction.

Write `recommendations.md`:

```markdown
# Recommendations — <slug>
_Refreshed <ISO-8601 UTC>. Quota 3/5 used. Next reset 2026-05-13T00:00Z (in 2h 17m)._

## RECOMMENDED — 2026-05-12-lgbm-cat-blend

- **CV (RMSE):** 0.7398 (std 0.0017)  ·  **Trust-adjusted:** 0.7381
- **Public LB last submitted:** 0.7423 (gap 0.0025, within 1× CV std → healthy)
- **Ideas:** `cv:stratified-kfold-5`, `feature:log1p_target`, `ensemble:lgbm-cat-blend`
- **Attribution:** `attr: jdoe123/eda-and-lgbm-baseline, msmith/catboost-blueprint`
- **Why this:** Best trust-adjusted CV. CV–LB gap is within 1 std. Diverse ideas — three independent models contributing.

## SAFE — 2026-05-12-lgbm-baseline-s42

- **CV (RMSE):** 0.7415 (std 0.0011)  ·  **Trust-adjusted:** 0.7404
- **Public LB last submitted:** 0.7411 (gap 0.0004)
- **Ideas:** `cv:stratified-kfold-5`, `feature:log1p_target`
- **Why this:** Smallest variance. Safer in private LB if the ensemble overfits.

## Other ranked candidates

| Rank | run_id | CV | std | Trust-adj | Public LB | Status |
|---|---|---|---|---|---|---|
| 3 | 2026-05-12-cat-baseline | 0.7432 | 0.0020 | 0.7412 | 0.7445 | completed |
| 4 | 2026-05-11-lgbm-with-tta | 0.7449 | 0.0008 | 0.7441 | n/a | completed |
| ⚠ | 2026-05-12-near-dup | 0.7388 | 0.0019 | 0.7369 | n/a | near_duplicate of #1 — excluded |

## How to submit one

Tell me, in your next message, something like:
- `"submit candidate 1"` (or `"submit 2026-05-12-lgbm-cat-blend"`)
- `"submit candidate 2 with message: <custom message>"` (attribution will be auto-appended)
- `"quota only"` (just refresh quota state, no submit)

Reminder: the final 2 submissions before deadline (Sun 2026-08-15 23:59 UTC) are **always**
user-picked, never auto-submitted (rules 3 and 9). I'll surface a separate
`final_selection.md` 6 hours before deadline.
```

Append `progress.jsonl` event `recommendation_refreshed`.

### Step 4 — Wait for user pick (or skip if invoked without --submit)

If invoked without `--submit`, exit 0 here. The recommendations file is the deliverable; the user reads it and triggers the next invocation with `--submit <run_id>`.

If invoked with `--submit <run_id>`, proceed to Step 5.

Append `progress.jsonl` event `awaiting_user_pick` only when the user has explicitly asked for recommendations without picking yet.

### Step 5 — Pre-submission gates

Before calling `kaggle competitions submit`:

1. **Run exists + complete?** `runs/<comp_slug>/stage2_modeling/runs/<run_id>/test_preds.csv` must exist; `attribution.md` must have a non-empty Citations section OR `+own` marker.
2. **Quota?** `remaining > 0`. If zero, write wait_until and exit (as in Step 2).
3. **Near-duplicate?** Run `near_duplicate.py` against all prior submissions in `submission_log.jsonl`. If found and the caller did not pass `--allow-near-duplicate`, escalate per rule 4.
4. **Attribution message?** Build the submission message. Template:
   ```
   <run_id> — CV <metric>=<score> (std <std>) | attr: <kernel1>, <kernel2>, <kernel3> [+ own]
   ```
   The `attr:` tokens come from `attribution.md`'s Citations section (max 3, sorted by count of distinct uses across the citations.bib's reference graph if that's tracked, else by first-appearance order). If the run has +own additions, append `+ own`.
5. **Deadline mode?** If `now > deadline - 6h`, refuse to auto-submit. Print the message but require the user to pass `--final-submission-confirm` to override.
6. **Pre-submit sanity (optional)** — if `run.yaml.external_tools.pre_submit_review: auto`, write a `handoff_prompts/pre_submit.md` for an external sanity-check (handoff #7) and proceed. If the response includes `no-go`, surface it in the next recommendations refresh.

If any gate fails, append `progress.jsonl` event `submit_blocked` with the reason and exit 3 (non-zero, recoverable).

### Step 6 — Submit

```bash
kaggle competitions submit -c <slug> \
  -f runs/<slug>/stage2_modeling/runs/<run_id>/test_preds.csv \
  -m "<message built in Step 5>"
```

Capture stdout — the Kaggle CLI prints a submission ID and a "submission file" path. Parse for the submission ID.

Append `submission_log.jsonl`:

```json
{"ts_utc": "...", "run_id": "...", "file": ".../test_preds.csv",
 "message": "<full message>", "kaggle_submission_id": "<id>",
 "public_lb": null, "cv_score": 0.7398, "cv_std": 0.0017,
 "candidate_rank_at_time": 1, "user_approved_by": "<user>"}
```

Update `quota_state.yaml`: `used_today += 1`, recompute `remaining`, `exhausted`.

Append `progress.jsonl` events `submitted` then `quota_used`.

### Step 7 — Poll for public LB

Public LB does not appear immediately. After ~30s, poll `kaggle competitions submissions -c <slug>` until the submission with our ID shows a `publicScore`. Backoff: 30s, 60s, 120s, 300s; give up after 10 minutes.

When public LB lands, update the `submission_log.jsonl` entry in place (rewrite the file with the updated entry — exception to append-only because this is a known-incomplete record). Update `leaderboard.csv` (`assets/leaderboard.py::record_submission_result`).

Append `progress.jsonl` event `public_lb_known` with `{run_id, public_lb, gap}`.

### Step 8 — Hand-off

Write `hand_off.md`:

```markdown
# Stage 3 → orchestrator hand-off (cycle <N>)

## What I did
- Refreshed recommendations.md with <N> ranked candidates.
- (Optional) Submitted <run_id> at <ts>. Public LB <score> (CV <score>, gap <delta>).

## What's true now
- Quota: <used>/<limit> used today. Remaining: <r>. Next reset: <ts>.
- Best public LB: <score> from <run_id>.
- Best trust-adjusted CV: <score> from <run_id>.
- CV–LB calibration health: <healthy | mild | suspect>.
- Days to deadline: <N>.

## What you should do next
- (If quota remaining): wait for the user to pick another candidate, OR
  delegate to Stage 2 for more modeling.
- (If exhausted): wait_until.txt is set; supervisor sleeps; loop the cycle on resume.
- (If approaching deadline): switch to deadline mode — Stage 2 should freeze
  new architectures and focus on ensembling existing OOFs.
```

Update `.heartbeat` and exit.

## Deadline mode

When `now > deadline - 24h`:

1. Refuse to invoke Stage 2 modeling for new architectures.
2. Refuse to submit without `--final-submission-confirm`.
3. Refresh recommendations as usual, but the top 2 slots become `FINAL CANDIDATE 1 (SAFE)` and `FINAL CANDIDATE 2 (AMBITIOUS)`, ranked by:
   - SAFE: lowest `cv_std` among the top 5 by trust-adjusted CV
   - AMBITIOUS: highest `trust_adjusted - cv_std` (i.e. takes the risk if CV is high enough)
4. The user picks both for final submission. The skill writes `final_selection.md`:
   ```markdown
   # Final selection — <slug>
   Picked at <ts> by <user>. Deadline <ts>.

   ## Submission slot 1 (SAFE)
   - run_id: <id>
   - CV: <score> (std <std>)
   - public LB (last seen): <score>
   - reasoning: <user paragraph>

   ## Submission slot 2 (AMBITIOUS)
   - run_id: <id>
   - CV: <score> (std <std>)
   - reasoning: <user paragraph>
   ```
5. The submissions themselves go through Steps 5–7 normally (with deadline-mode flags set).

## What this stage does NOT do

- Decide what to submit (the user does, after reading recommendations).
- Train new models (Stage 2's job).
- Pull more recon (Stage 1's job).
- Edit the test predictions in any way — `test_preds.csv` from Stage 2 goes to Kaggle byte-for-byte.

## When to load which reference

| File | Load when |
|---|---|
| `references/ranking-rubric.md` | Step 3 — building the recommendation ranking |
| `references/quota-reconciliation.md` | Step 1 — parsing `kaggle competitions submissions` |
| `references/final-selection.md` | Deadline mode |
| `auto-kaggle/references/state-contract.md` | Always |
| `auto-kaggle/references/integrity-rules.md` | Steps 5–6 (rules 2, 3, 4, 6, 9) |
| `auto-kaggle/references/kaggle-cli-basics.md` | Steps 1, 6, 7 |
| `auto-kaggle/references/long-running-protocol.md` | Quota exhaustion / resume |
| `auto-kaggle/references/escalation-policy.md` | Quota desync, near-duplicate override, deadline mode |
