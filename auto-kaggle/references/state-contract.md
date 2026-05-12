# State contract — `runs/<comp_slug>/`

Single source of truth for every file the four stage skills read and write. **No agent memory crosses invocations.** If a piece of information is not on disk under `runs/<comp_slug>/`, it does not exist.

`<comp_slug>` is the Kaggle competition slug exactly as it appears in the URL (e.g. `titanic`, `playground-series-s4e5`, `child-mind-institute-detect-sleep-states`). Never invent a slug — always read it from the user-supplied URL or from `kaggle competitions list`.

## Full layout

```
runs/<comp_slug>/
├── run.yaml                       # comp slug, compute env, target tier, deadline, supervisor mode
├── .heartbeat                     # JSON: {stage, substep, ts_utc, pid}  — overwritten every tick
├── progress.jsonl                 # append-only: every meaningful sub-step (see schema below)
├── STOP                           # if present, supervisor and orchestrator exit cleanly
├── PAUSE                          # if present, orchestrator finishes current substep then idles
├── data/                          # raw + processed competition data (gitignored)
│   ├── raw/                       # output of `kaggle competitions download`
│   └── processed/                 # cleaned / feature-engineered artifacts
├── stage0_bootstrap/
│   ├── comp_profile.yaml          # task type, metric, submission format, daily quota, deadline
│   ├── rules_summary.md           # human-readable distillation of Overview + Rules + Evaluation
│   ├── data_stats.md              # row counts, column types, missingness, target distribution
│   ├── compute_env.yaml           # user-chosen env: kaggle-notebook | local-gpu | cloud-gpu | cpu-only + specs
│   └── hand_off.md                # one paragraph briefing for the next stage
├── stage1_recon/
│   ├── kernels_index.json         # top public kernels: id, author, votes, public_lb, last_updated
│   ├── kernels/                   # downloaded notebooks, one .ipynb per kernel (reference only)
│   ├── ideas_pool.md              # distilled techniques with per-kernel attribution
│   ├── citations.bib              # BibTeX-style entries for every referenced kernel
│   ├── last_recon_at              # ISO-8601 UTC timestamp of last recon pull
│   └── hand_off.md
├── stage2_modeling/
│   ├── pipeline.py                # the user-owned pipeline (re-implemented, not forked)
│   ├── cv_split.yaml              # CV scheme + rationale (group/time/stratified)
│   ├── runs/<run_id>/             # one folder per training run
│   │   ├── config.yaml
│   │   ├── oof.npy                # out-of-fold predictions on train set
│   │   ├── test_preds.csv         # predictions on test set in submission format
│   │   ├── cv_score.json          # per-fold + aggregate CV metric
│   │   ├── attribution.md         # which recon ideas were used + own additions
│   │   └── train.log
│   ├── leaderboard.csv            # rolling table: run_id, cv_score, public_lb, gap, status
│   └── hand_off.md
└── stage3_submit/
    ├── submission_log.jsonl       # append-only: every actual Kaggle submission
    ├── quota_state.yaml           # used_today, daily_limit, last_reset_utc, next_reset_utc
    ├── wait_until.txt             # ISO-8601 UTC — supervisor sleeps until this time
    ├── recommendations.md         # ranked candidates with reasoning, refreshed each cycle
    ├── final_selection.md         # the 2 final submissions the user picked (deadline-side)
    └── hand_off.md
```

## File-by-file specs

### `run.yaml` (created at Stage 0, never rewritten by later stages)

```yaml
comp_slug: child-mind-institute-detect-sleep-states
url: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states
created_at_utc: 2026-05-12T08:30:00Z
deadline_utc: 2026-08-15T23:59:00Z
target_tier: silver-floor-gold-ceiling   # silver | silver-floor-gold-ceiling | gold
kaggle_username: <user's kaggle handle>   # used in attribution
supervisor:
  mode: claude-loop | shell-supervisor | manual
  poll_seconds: 1800                       # how often supervisor re-invokes orchestrator
  max_runtime_hours: null                  # null = run until deadline or STOP
```

### `.heartbeat` (overwritten every tick — read by user to check status without invoking agent)

```json
{
  "stage": "stage2_modeling",
  "substep": "training fold 3/5 of run 2026-05-12-lgbm-baseline",
  "ts_utc": "2026-05-12T14:22:11Z",
  "pid": 47215,
  "agent": "claude-opus-4-7"
}
```

### `progress.jsonl` (append-only, one JSON object per line)

Every meaningful sub-step appends a line:

```json
{"ts_utc": "2026-05-12T08:31:02Z", "stage": "stage0", "event": "comp_profile_written"}
{"ts_utc": "2026-05-12T08:32:14Z", "stage": "stage0", "event": "data_downloaded", "bytes": 1432165120}
{"ts_utc": "2026-05-12T08:45:00Z", "stage": "stage1", "event": "recon_pulled", "kernels": 27}
{"ts_utc": "2026-05-12T09:10:33Z", "stage": "stage2", "event": "run_started", "run_id": "2026-05-12-lgbm-baseline"}
{"ts_utc": "2026-05-12T09:45:11Z", "stage": "stage2", "event": "fold_done", "run_id": "...", "fold": 0, "score": 0.7421}
{"ts_utc": "2026-05-12T11:02:55Z", "stage": "stage3", "event": "submitted", "file": "ens_v3.csv", "public_lb": 0.7488}
{"ts_utc": "2026-05-12T11:03:01Z", "stage": "stage3", "event": "quota_used", "used": 3, "limit": 5}
```

Resume protocol: orchestrator reads the **last** line per stage to know where to pick up. Never trust line ordering across stages — always sort by `ts_utc`.

### `stage0_bootstrap/comp_profile.yaml`

```yaml
task_type: time-series-event-detection   # canonical values defined in
                                         # auto-kaggle-bootstrap/references/task-type-detection.md
                                         # (tabular-binary | tabular-multiclass | tabular-regression
                                         # | tabular-ranking | image-classification | image-segmentation
                                         # | image-detection | image-instance-segmentation
                                         # | nlp-classification | nlp-regression
                                         # | nlp-token-classification | nlp-generation
                                         # | time-series-forecast | time-series-event-detection
                                         # | audio-classification | multimodal | graph
                                         # | recommendation | other)
metric:
  name: event_detection_ap
  direction: maximize
  better_when: higher
submission:
  format: csv                            # csv | code-only-notebook | code-only-script
  columns: [series_id, event, step, score]
  size_limit_mb: null
  code_only: false
quota:
  daily_limit: 5
  resets_at_utc: "00:00"
team:
  max_size: 5
  solo_required: false
external_data:
  allowed: true
  must_be_shared: true                   # if true, all external data must be on Kaggle datasets
```

### `stage3_submit/submission_log.jsonl` (append-only, never edited)

```json
{"ts_utc": "2026-05-12T11:02:55Z", "file": "stage2_modeling/runs/.../test_preds.csv",
 "message": "lgbm 5fold seed=42 + augment_v2 + citation:kernel/jdoe123-v7",
 "kaggle_submission_id": "12345678",
 "public_lb": 0.7488, "cv_score": 0.7421,
 "candidate_rank_at_time": 1,
 "user_approved_by": "<user>"}
```

### `stage3_submit/quota_state.yaml` (overwritten after each submission)

```yaml
used_today: 3
daily_limit: 5
last_submission_utc: 2026-05-12T11:02:55Z
last_reset_utc: 2026-05-12T00:00:00Z
next_reset_utc: 2026-05-13T00:00:00Z
remaining: 2
exhausted: false
```

When `exhausted: true`, write `wait_until.txt` with `next_reset_utc` and exit.

## Read-order on resume

When `auto-kaggle resume <comp_slug>` is invoked, the orchestrator reads in this order:

1. `run.yaml` — confirms the comp exists and gets supervisor config.
2. `STOP` / `PAUSE` — if present, exit immediately or idle.
3. `progress.jsonl` — last `ts_utc` per stage, decides "where we are".
4. `stage3_submit/wait_until.txt` — if exists and `now < wait_until`, sleep / yield to supervisor.
5. `stage3_submit/quota_state.yaml` — figure out today's budget.
6. Latest `hand_off.md` of the most-advanced stage — the briefing for what to do next.

If any contract file is missing on resume, the orchestrator escalates to the user **instead of guessing** — see `escalation-policy.md`.

## Hand-off file convention

Every stage's `hand_off.md` answers exactly three questions, in three short paragraphs, in this order:

1. **What I did** — the new artifacts I wrote, with paths.
2. **What's true now** — the facts the next stage should act on (best CV score, current public LB, ideas to try next, quota remaining, blocking issues).
3. **What you should do next** — concrete next action, framed as a directive to the next stage skill.

No prose-y narration. The next stage reads only `hand_off.md` + the structured files it lists.
