# Escalation policy — when to stop and ask the human

A long-running auto-grinder is dangerous precisely when it confidently does the wrong thing. The right reflex is to **pause and write a question to the user**, not to guess.

This file lists the escalation triggers. Each stage skill must check them; the orchestrator enforces them at every hand-off.

---

## Hard escalations (skill stops and waits)

The skill **does not proceed** on its own. It writes to `stage3_submit/recommendations.md` (or the relevant stage's hand_off) and exits.

| Trigger | Reason |
|---|---|
| Final 2 submission selection (deadline-side) | Rule 3 — user must pick. Skill ranks; user decides. |
| Submitting in the last 6 hours before deadline | High-cost, no-takebacks. User confirms each one. |
| About to submit a candidate flagged for near-duplicate (Rule 4) | Possible LB probing — needs justification. |
| Any change to `cv_split.yaml` after Stage 0 | Rule 7 — only the user changes the CV scheme. |
| Stale heartbeat from another agent (`pid` exists and is owned by another active process) | Possible concurrent run — refuse to mutate state. |
| `quota_state.yaml` disagrees with `kaggle competitions submissions` by more than 1 | State corruption — user reconciles. |
| Competition rules contain words like "team merge frozen", "private sharing", "data leak rumored" in the latest pull | Comp may be in flux — confirm before continuing. |
| Two consecutive submissions move public LB by more than 5× the median run-to-run delta | Either a breakthrough or a leak; either way, user reviews. |
| External tool handoff returns content that names a file outside `runs/<comp_slug>/` | External tool tried to write outside its sandbox. |
| Compute env detection disagrees with prior run (e.g. previously local-gpu, now cpu-only) | Reproducibility risk — user confirms. |

---

## Soft escalations (skill continues but flags)

The skill proceeds, but appends a `⚠` line to the next `hand_off.md`:

| Trigger | Flag |
|---|---|
| CV–LB gap exceeds threshold for the last 3 runs | "CV/LB miscalibrated — investigate before final pick" |
| Recon pull returns 0 new kernels for 2 cycles in a row | "Public LB has stabilized — switch to ensembling" |
| Single training run exceeds estimated budget by 1.5× | "Budget overrun — review estimator" |
| External data source listed but Kaggle dataset ID is empty | "External data needs a Kaggle dataset ID before submit" |
| Deadline less than 7 days and no submission has cleared `cv_score` median of public top-20 | "Behind pace for target tier — reassess strategy" |

---

## How to escalate (operational)

In `recommendations.md` (or the relevant hand_off), append a section:

```markdown
## ESCALATION — <short title>

**Trigger:** <which rule from this file>
**Observed:** <the facts that triggered it, with file paths>
**Default if I proceed:** <what would happen without user input>
**Question for you:** <one specific question with the options labeled (a) (b) (c)>

I will not act on this until I see one of:
- <file path or sentinel> updated with one of the options, OR
- The `PAUSE` file in this run is removed and you give me a fresh instruction.
```

Then write `progress.jsonl` event `escalated` with the trigger name and exit.

The orchestrator on resume sees the escalation, prints it again, and continues to exit. Only an explicit user reply (a hand-edited file, a new prompt naming the option) clears it.

---

## What is **not** an escalation

Don't escalate for things that can be auto-handled:

- A training run failing — log it, retry with a smaller config, append `progress.jsonl` event `run_failed`. Only escalate after 3 consecutive failures on the same config.
- A `kaggle competitions submit` returning a transient HTTP error — retry up to 3 times with exponential backoff, then escalate.
- A recon pull returning a kernel with malformed metadata — skip that kernel, log it, continue.
- The user being asleep — they will wake up and read `.heartbeat` and `recommendations.md`. The skill is allowed to keep training in the meantime.

Escalation is for things that **cost the user something if you guess wrong**, not for normal noise.
