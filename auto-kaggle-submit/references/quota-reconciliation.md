# Quota reconciliation

Stage 3's first job on every invocation. The local `quota_state.yaml` is a *cache*; Kaggle's `kaggle competitions submissions -c <slug>` is the *truth*. Drift between them is the leading cause of "I used my last submission and the skill thought I had one left."

## The reconciliation procedure

```python
from datetime import datetime, timezone

def reconcile_quota(comp_slug, run_dir, daily_limit):
    raw = subprocess.check_output(["kaggle", "competitions", "submissions",
                                   "-c", comp_slug, "-v"], text=True)
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    used_today = 0
    for line in raw.splitlines():
        line = line.strip()
        # Each row starts with a YYYY-MM-DD-HH:MM:SS timestamp in the first column.
        if len(line) >= 10 and line[:10] == today_utc and not line[10].isalnum():
            used_today += 1
    # Account for build-failed submissions: they do count against quota.
    # The -v output includes failed builds.

    cached = load_quota_state(run_dir / "stage3_submit/quota_state.yaml")
    discrepancy = abs(used_today - (cached.used_today if cached else 0))
    if discrepancy > 1:
        escalate("quota desync", observed=used_today, cached=cached.used_today)
        # Take Kaggle's count as truth and proceed under escalation.

    state = QuotaState(
        used_today=used_today,
        daily_limit=daily_limit,
        remaining=max(0, daily_limit - used_today),
        last_submission_utc=parse_last_submission_ts(raw),
        last_reset_utc=today_midnight_utc(),
        next_reset_utc=tomorrow_midnight_utc(),
        exhausted=(used_today >= daily_limit),
    )
    write_quota_state(run_dir / "stage3_submit/quota_state.yaml", state)
    return state
```

## What "today UTC" means

`datetime.now(timezone.utc).date()`. Kaggle's daily quota resets at exactly 00:00:00 UTC, regardless of the user's local time. If your wall clock says 2:00 AM Beijing time, that's 18:00 UTC the previous day — your quota does NOT reset until 8:00 AM Beijing time.

The skill **only** uses UTC for quota math. Display can be in local time as a courtesy, but the canonical state is UTC.

## Failed submissions count

Kaggle counts a submission against your daily quota the moment it accepts the upload, even if the eventual scoring fails (build error, runtime error in a code-only comp, etc.). So `used_today` includes:

- Successful submissions with a public LB
- Submissions still pending scoring
- Submissions that failed scoring

This is why Kaggle's `kaggle competitions submissions -v` is the truth: it knows about all three. A local count based on "I submitted this many times" can be wrong if a CLI call appeared to fail but actually got accepted server-side.

## Parsing the `kaggle competitions submissions` output

The `-v` (verbose) flag shows additional columns. The first column is always the submission timestamp in `YYYY-MM-DD HH:MM:SS` (UTC). The format is **not** stable JSON — it's a tabular text dump:

```
fileName    date                  description           status    publicScore   privateScore
submi…csv   2026-05-12 14:22:03   v1 baseline          complete   0.7488        ...
submi…csv   2026-05-12 09:15:11   v1 EDA               complete   0.7411        ...
submi…csv   2026-05-11 21:03:55   prior day            complete   0.7401        ...
```

Defensive parsing:

- Skip header lines (no leading digit).
- Skip empty lines.
- The first 10 characters of a data row are the date. If they match `today_utc`, count.
- The submission ID is **not** in this output by default. If we need it (for poll-public-LB), use the `--csv` variant or scrape the `fileName` + `date` to disambiguate.

If parsing returns 0 rows from a non-empty output, escalate — the format may have drifted in a newer Kaggle CLI version.

## Tracking submissions made by teammates

If `run.yaml.team` lists multiple members, the agent treats `kaggle competitions submissions` as showing **only the current user's** submissions. Teammates' submissions affect the team's daily quota but Kaggle counts per-user, and the team's effective quota is the union.

For team competitions:

- `daily_limit_per_user`: typically 5 per user.
- `daily_limit_per_team`: typically 5 (team-wide), but check rules — some comps allow more per team.
- The skill reconciles only the current user's count, and **flags** in `recommendations.md` if the team has used more than expected (teammate's actions). The user resolves manually.

For solo competitions, this is moot — there is no team.

## `wait_until.txt` semantics

When `state.exhausted == True`:

```python
wait_until_path = run_dir / "stage3_submit/wait_until.txt"
write_atomic(wait_until_path, state.next_reset_utc.isoformat() + "Z\n")
```

The supervisor reads this on every poll. If `now < wait_until`, it sleeps until then (`sleep <delta_seconds>`) without invoking Claude again. This is the main mechanism for "skill goes quiet for hours" → "supervisor wakes Claude exactly when quota refreshes."

When `now >= wait_until`:

- The supervisor wakes Claude with `/auto-kaggle resume <slug>`.
- The orchestrator's Step 4 of resume reads `wait_until.txt`, sees `now >= wait_until`, deletes the file.
- Stage 3 then reconciles quota (which now reads `used_today: 0`) and continues normally.

The file is in the **submit stage's** directory because that's the stage that wrote it. The orchestrator doesn't manage it; only Stage 3 creates/deletes it.

## Edge case: clock skew

If `now < state.last_submission_utc` (system clock went backwards):

- Refuse to compute quota until the clock is fixed.
- Escalate immediately. This is a system problem, not a Kaggle problem.

If `now < state.next_reset_utc` and we expected a reset (e.g. wait_until was 1h in the past):

- The host machine slept past the wait_until. That's fine — proceed to reconcile.
- Append `progress.jsonl` event `quota_reset_late` with the actual lag.

## Edge case: Kaggle CLI says fewer submissions than we cached

If `used_today < state.used_today`:

- This means Kaggle did not count a submission that we thought we made. Possibilities:
  - Submission was rejected at upload (network blip, file too large, format invalid). Kaggle didn't count it; we logged it.
  - Race condition where our log write happened before Kaggle's count updated.
- Take Kaggle's count as truth (lower is better for us — gives us back a slot). Log `progress.jsonl` event `quota_cache_too_high`.
- Inspect the most recent `submission_log.jsonl` entries; if any have `kaggle_submission_id: null` and `public_lb: null`, mark `status: rejected_at_upload`.

## Edge case: rules say daily limit is different from default

Stage 0 should have extracted the daily limit into `comp_profile.quota.daily_limit`. If that's missing, default to 5. If it's a weird value like 2 or 10, trust the bootstrap parse.

Some comps have rule changes mid-competition. If the user reports "quota changed", the skill re-bootstraps (Stage 0 with `--re-bootstrap`) and re-reads `daily_limit`.

## Final-selection quota

Kaggle gives each team `final_selections` (default 2) picks at deadline — these are the submissions that count for private LB. The picking is **not** a submission; it's a UI action ("select for final") that the user does on the Kaggle website. The skill does NOT do this automatically.

In `recommendations.md` near deadline, the skill produces a `final_selection.md` with the proposed two submissions. The user goes to the Kaggle website and selects them via the UI. The skill records `final_picked_at_utc` in the log.

There is no API for "select this as final" — it's a deliberate Kaggle design to keep humans in the loop.
