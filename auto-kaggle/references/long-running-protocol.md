# Long-running protocol

A Kaggle medal run takes **days to months**, not minutes. The agent process will be interrupted: context window fills up, Claude Code session ends, the host machine reboots, the network drops mid-`kaggle competitions submit`. The skill is designed so that **any interruption is recoverable** and the next invocation resumes where the last one left off.

This document defines exactly how that works. Every stage skill is required to obey it.

---

## Core principles

1. **All state on disk.** The agent owns nothing between invocations. Everything needed to resume is in `runs/<comp_slug>/` (see `state-contract.md`).
2. **Idempotent micro-steps.** Re-running any sub-step with the same inputs must produce the same outputs, or detect existing outputs and skip. No "I half-trained fold 2 and now restarting overwrites it" failure mode.
3. **Append-only logs.** `progress.jsonl` and `submission_log.jsonl` are never rewritten — only appended to. Crashes mid-write leave at worst a truncated last line, which a resume parser must tolerate.
4. **Atomic file writes.** Critical state files (`comp_profile.yaml`, `quota_state.yaml`, `.heartbeat`) are written via `write to <path>.tmp` → `rename <path>.tmp <path>`. Never write in place.
5. **Resume by default.** `/auto-kaggle <comp_slug>` always means *resume*. A fresh start requires explicit `--restart` and overwrites only after confirming with the user.

---

## Heartbeat

Every stage skill, while active, overwrites `runs/<comp_slug>/.heartbeat` at least once every 60 seconds with:

```json
{"stage": "...", "substep": "...", "ts_utc": "...", "pid": <int>, "agent": "..."}
```

This file is **not** read by the agent itself. It exists so the user can `cat runs/<comp_slug>/.heartbeat` from any terminal and see what's happening without waking the agent.

Supervisors (below) read the heartbeat too — if `ts_utc` is older than 5 minutes and the process is still claimed to be running, the supervisor assumes a crash and re-invokes the agent.

---

## Progress log (`progress.jsonl`)

Append-only. One line per meaningful sub-step. Schema is in `state-contract.md`. Every stage skill emits these events:

| Stage | Event | Triggers |
|---|---|---|
| stage0 | `comp_profile_written` | after `comp_profile.yaml` is finalized |
| stage0 | `data_downloaded` | after `kaggle competitions download` finishes |
| stage1 | `recon_started` | beginning of a recon pull |
| stage1 | `recon_pulled` | end of a successful pull, with `kernels` count |
| stage1 | `idea_extracted` | per idea added to `ideas_pool.md` |
| stage2 | `run_started` | new training run created |
| stage2 | `fold_done` | each completed fold, with score |
| stage2 | `run_finished` | training run complete |
| stage3 | `recommendation_refreshed` | `recommendations.md` rewritten |
| stage3 | `awaiting_user_pick` | skill is paused for user to pick a submission |
| stage3 | `submitted` | after successful `kaggle competitions submit` |
| stage3 | `quota_used` | after quota counter ticks |
| stage3 | `quota_exhausted` | when `used_today == daily_limit`, `wait_until.txt` was written |

On resume, the orchestrator reads the **last** event per stage (sorted by `ts_utc`) and dispatches:

- Latest event is `quota_exhausted` and `now < wait_until` → exit with code 0 (supervisor handles sleep).
- Latest event is `awaiting_user_pick` → print the current `recommendations.md` and exit waiting for user.
- Latest event is mid-stage (e.g. `fold_done` with `fold < n_folds - 1`) → continue that run from the next fold.
- Otherwise → proceed to the next stage per the normal flow.

---

## Wait-until protocol (quota exhaustion)

When `stage3_submit` detects `used_today == daily_limit`:

1. Compute `next_reset_utc` (the next `00:00 UTC`).
2. Write `runs/<comp_slug>/stage3_submit/wait_until.txt` containing exactly the ISO-8601 timestamp, e.g. `2026-05-13T00:00:00Z\n`.
3. Append `progress.jsonl` event `quota_exhausted`.
4. Print to stdout a single line: `WAIT_UNTIL 2026-05-13T00:00:00Z (in 2h 17m)`.
5. Exit with code 0.

The next invocation:

- If `wait_until.txt` exists and `now < wait_until` → print `STILL_WAITING …`, exit 0.
- If `wait_until.txt` exists and `now >= wait_until` → delete the file, reset `quota_state.yaml` counters, append `progress.jsonl` event `quota_reset`, proceed.

While waiting, the skill can still do **non-submission** work: more recon, more training, more ensembling. Quota-exhausted means stage 3 sleeps, **not** stages 1 / 2.

---

## Stop / pause sentinels

Two sentinel files give the user manual override:

- **`runs/<comp_slug>/STOP`** — orchestrator and supervisor both exit cleanly on next check (after finishing the current atomic write). Once present, the run is permanently paused; remove the file to resume.
- **`runs/<comp_slug>/PAUSE`** — orchestrator finishes its current sub-step, then idles in a check loop with 60s polling. Removing the file resumes.

Both checks happen at the top of every micro-step. The skill never ignores them.

---

## Supervisor (outer scheduling)

A skill is invoked, not always-on. To get genuinely autonomous multi-day operation, the orchestrator must be driven from outside Claude Code's interactive session. Three supported modes (`run.yaml.supervisor.mode`):

### Mode A: `claude-loop` (Claude Code `/loop`)

Inside Claude Code, the user runs:

```
/loop /auto-kaggle resume <comp_slug>
```

The `/loop` slash command re-fires the prompt at intervals (or self-paced). Pros: stays inside Claude Code, uses your existing auth and context. Cons: stops when you close Claude Code.

### Mode B: `shell-supervisor` (assets/supervisor.sh)

Shipped at `auto-kaggle/assets/supervisor.sh`. Runs outside Claude Code as a background process:

```bash
nohup bash auto-kaggle/assets/supervisor.sh <comp_slug> > supervisor.log 2>&1 &
```

It loops:

1. Check `STOP` file → exit if present.
2. Read `.heartbeat` → if older than 5 minutes, assume crash.
3. Read `wait_until.txt` → if `now < wait_until`, `sleep` until it.
4. Invoke `claude -p "/auto-kaggle resume <comp_slug>"` (or `codex run "..."`) in headless mode.
5. After the invocation returns, sleep `run.yaml.supervisor.poll_seconds` (default 1800).
6. Repeat.

Pros: survives Claude Code being closed. Cons: needs CLI auth set up for headless mode and a host that stays up.

### Mode C: `manual`

User invokes the orchestrator by hand whenever they remember. The skill still resumes correctly, but no automatic scheduling. This is the safe default until the user has watched a few cycles and is comfortable letting it run unattended.

---

## Crash recovery contract

Every stage skill must tolerate:

- **Partial file writes.** A `.tmp` file from a prior run is detected and deleted, never read.
- **Truncated last line of `progress.jsonl`.** The parser drops the last line if it does not parse as JSON, logs `progress.jsonl tail dropped: <bytes>`, and continues.
- **Stale heartbeat.** If `.heartbeat` says `pid: X` but `X` is not running, the orchestrator overwrites it and proceeds. Never assume another agent is still alive based on heartbeat alone.
- **Quota state out of sync with Kaggle.** On resume, the orchestrator runs `kaggle competitions submissions -c <slug>` to reconcile `quota_state.yaml` with reality before submitting again. Kaggle's count is authoritative; ours is a cache.

If any of these conditions cannot be safely auto-recovered, the orchestrator **escalates** (see `escalation-policy.md`) and exits — never guesses.
