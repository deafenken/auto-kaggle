# Throttle and rate-limit handling

Stage 1 fires the Kaggle CLI more than any other stage. This file is the rule set for not getting blocked, and not wasting cycles re-pulling unchanged kernels.

## The interval (`run.yaml.recon_interval_hours`)

Default: `6` (four cycles per day). Reasonable range: `2` to `24`.

| Days to deadline | Suggested interval | Reasoning |
|---|---|---|
| > 30 | 12 h | Kernels move slowly; daily cadence is fine |
| 14–30 | 6 h (default) | Top kernels are getting refreshed; catch updates within a day |
| 7–14 | 4 h | Movement accelerates as more competitors join |
| 3–7 | 2 h | Final week is when new top-public-LB kernels appear |
| < 3 | 1 h, but cap kernel pulls per cycle at 5 | Mostly looking for last-minute leaks the leaderboard exposes |

The interval lives in `run.yaml` and the user can override it at any time by editing the file. The skill reads it fresh on every invocation.

## The throttle check (Step 0 of recon)

```python
import json, time
from pathlib import Path
from datetime import datetime, timedelta, timezone

last_recon_path = run_dir / "stage1_recon" / "last_recon_at"
interval_h = run_cfg.get("recon_interval_hours", 6)
now = datetime.now(timezone.utc)

if last_recon_path.exists() and not force:
    last = datetime.fromisoformat(last_recon_path.read_text().strip().replace("Z", "+00:00"))
    age_h = (now - last).total_seconds() / 3600
    if age_h < interval_h:
        print(f"recon throttled — last pull was {age_h:.1f}h ago, next allowed in "
              f"{interval_h - age_h:.1f}h")
        return 0
```

The supervisor still wakes the orchestrator on its `poll_seconds` cadence; the throttle is *inside* recon, not in the supervisor. This way the orchestrator can still do non-recon work (modeling, submit) on those wake-ups.

## Per-call rate-limit handling

`kaggle_helpers._kaggle` already implements backoff on 429:

- 1st 429: sleep 60s, retry
- 2nd: sleep 5min, retry
- 3rd: sleep 30min, retry
- After 3rd: raise `KaggleRateLimit`

Recon catches the raise and exits with code 0 — the supervisor's next poll will retry. Append `progress.jsonl` event `recon_rate_limited` so the user can see it.

## Spacing within a cycle

Even when not rate-limited, space CLI calls by ~1 second to be a good citizen:

```python
for i, kernel_ref in enumerate(to_pull):
    pull_kernel(...)
    if i + 1 < len(to_pull):
        time.sleep(1)
```

For very large pull batches (rare — capped at 20 per cycle, see SKILL.md Step 3), the total time is ~30s which is comfortably below any plausible rate limit window.

## Cap per cycle

Hard cap of `20` new + updated kernels per cycle. If more remain:

1. Sort the remaining by `votes_normalized = votes / (max_votes + 1) + 0.5 * has_public_lb`.
2. Pull the top 20, leave the rest for the next cycle.
3. Append `progress.jsonl` event `recon_pull_capped` with the deferred count.
4. They get pulled on the next cycle automatically (they remain marked `new` until pulled).

This prevents one cycle from monopolizing the supervisor.

## "Time freeze" protocol

If the host machine's clock is wrong (e.g. just rebooted, NTP not synced yet), the throttle math breaks. Sanity check at the top of Step 0:

```python
if last_recon_path.exists():
    last_ts = ...
    if now < last_ts:
        # Clock went backwards — refuse, escalate.
        print(f"system clock is before last_recon_at ({now} < {last_ts}); refusing")
        return 3
```

This is an escalation, not a retry. Time misconfigurations cascade silently into bad quota state, so we fail loud.

## Why we pull both `voteCount` and `scoreDescending`

The two listings overlap but not entirely:

- `voteCount` surfaces educational / well-documented kernels that may not be the public-LB tops.
- `scoreDescending` surfaces the actual current public-LB tops, including freshly-published "blind blend" kernels that have no votes yet.

A submission strategy that only looks at one is incomplete. Stage 2's recommendations are a function of both: a high-vote kernel suggests well-vetted techniques; a high-LB-no-votes kernel suggests where the LB is moving *right now*.

## What we explicitly do **not** scrape

- Discussion threads — not yet. Could be added later via external-tools handoff #1.
- User profiles. Not relevant.
- Leaderboard pages. We use `kaggle competitions submissions -c <slug>` in Stage 3 instead.
- Private kernels (not visible to us).
- Comments on kernels (low signal-to-noise).

## When the rate limit story changes

Kaggle's documented rate limits are vague. If Kaggle's CLI starts returning 429s consistently:

1. Lengthen the per-call spacing to 3–5s.
2. Drop the cap from 20 to 10 per cycle.
3. Increase the interval.
4. If still hitting 429, escalate to the user — there might be an account-level issue.

Never circumvent the limit by signing in with a different account (Rule 5).
