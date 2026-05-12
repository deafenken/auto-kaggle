# Auto Kaggle Skills

`auto-kaggle` is a staged "give me a Kaggle URL → grind for medals across days" skill suite for Claude Code and Codex agents.

The core idea: split Kaggle medal hunting into 4 stages, each owned by its own skill, with stages handing off through files under `runs/<comp_slug>/`. **All state lives on disk**, so the agent process can crash and resume at will until the competition deadline (or until you drop a `STOP` file).

---

## TL;DR

> You give it a Kaggle competition URL. It asks for your compute environment, downloads data, periodically scrapes top public kernels for ideas (with attribution), trains its own CV-aware pipeline, shows you real-time quota ("3/5 submissions used today, next reset in 2h 17m"), and presents ranked submission candidates for **you** to pick — then runs `kaggle competitions submit` for the one you chose and goes back to training.

It is **not** a cheat tool, **not** an auto-submit firehose, **not** a "blindly fork the top public kernel" script. The final 2 submissions are always your call.

---

## Beginner's Notice

### Who this is for

- You have a Kaggle account, can log in, and can accept competition rules on the website.
- You want to lock a silver medal / reach for gold (the default tier is `silver-floor-gold-ceiling`).
- You have at least one machine that can stay on for a long time (local GPU, cloud GPU, or an open Kaggle Notebook). Without that, "multi-day fully automatic" is moot.
- You're willing to check `recommendations.md` once or twice a day and pick the candidate to submit.

### Who this is **not** for

- People who have never seen a Kaggle competition. Run `titanic` end-to-end first by hand.
- Knowledge / Tutorial competitions with no medals — the bootstrap stage rejects these.
- People wanting to bypass submission quotas, multi-account, or strip attribution from forked kernels. Integrity rules 1 / 2 / 5 block all of this.

### What you need to set up first

1. **Install Claude Code or the Codex CLI**, and confirm normal conversation works.
2. **Install + authenticate the Kaggle CLI:**
   ```bash
   pip install --upgrade kaggle
   # Download kaggle.json from https://www.kaggle.com/<you>/account
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   kaggle competitions list   # if this lists comps, you're good
   ```
3. **Open the competition in a browser, log in, and click the "Submit" / "Late Submission" button to accept the rules**. This step is not automatable — if you skip it, `kaggle competitions download` returns 403.
4. **Decide on a compute environment** (bootstrap will ask you to pick):
   - `kaggle-notebook` — free T4×2 / P100, 9h per kernel, 30h GPU/week. Mandatory for code-only comps.
   - `local-gpu` — your own machine; supply GPU model and VRAM.
   - `cloud-gpu` — Colab Pro / Lambda / Vast / RunPod; you keep it alive.
   - `cpu-only` — only viable for small tabular comps.
5. **Vocabulary you should know**:
   - **public LB vs private LB**. Public LB scores a slice of the test set in real time. Private LB is the rest, revealed after deadline; medals come from private LB. **Chasing public LB blindly is the classic shake-up trap.**
   - **shake-up** — large public→private rank movement; top-10 public dropping to 200+ on private is routine.
   - **CV (cross-validation)** — your local score; the only signal that actually predicts private LB.
   - **daily quota** — usually 5 submissions/day, resets at 00:00 UTC.
   - **attribution** — your submission message must name the public kernels that inspired this submission. The skill enforces this.
6. **Read `auto-kaggle/references/integrity-rules.md`**, especially rules 1 / 3 / 4 / 5 — they're what stops your account from being reported.

### If you're new to Kaggle (the most important step — do this first)

No amount of skill scaffolding makes up for not knowing Kaggle culture. Before you point this at a real medal target:

1. **Walk through Titanic by hand using the skill.** `auto-kaggle titanic` runs the full 4-stage flow on a no-medal comp so you can feel how `recommendations.md`, quota tracking, and the stage hand-offs work. No risk.
2. **Pick a medium-popularity ongoing comp** (< 1000 teams, 1+ month to deadline; Playground Series competitions are good first targets). Let the skill run for a week and watch:
   - Is the gap between CV and public LB stable?
   - Are the top candidates in `recommendations.md` actually competitive?
   - How much does your manual final pick differ from the skill's #1 ranked?
3. Only when you can explain in a few sentences "what the top public kernels here are doing, how private LB is likely to shake, and what I can add to beat them" should you point the skill at a comp you actually want to medal in.

This step matters more than any tooling below. Skip it and you're just burning GPU hours and submission slots.

---

## What each of the four skills does

| Skill | Role | Input | Outputs |
|---|---|---|---|
| `auto-kaggle` | Orchestrator (no research, only routing + gating) | Comp URL + 4 user answers | `runs/<slug>/run.yaml`, heartbeat, progress.jsonl, hand-off integrity checks |
| `auto-kaggle-bootstrap` | Stage 0: parse comp, download data, ask compute env | comp URL | `comp_profile.yaml` / `rules_summary.md` / `data_stats.md` / `compute_env.yaml` |
| `auto-kaggle-recon` | Stage 1: periodically scrape top public kernels, distill ideas | comp_profile + last_recon_at | `kernels_index.json` / `ideas_pool.md` / `citations.bib` |
| `auto-kaggle-modeling` | Stage 2: build the user's own pipeline with CV; integrate recon ideas | ideas_pool + comp_profile | `pipeline.py` / `runs/<run_id>/` / `leaderboard.csv` |
| `auto-kaggle-submit` | Stage 3: rank candidates, track quota, recommend, submit on user pick | leaderboard + quota_state | `recommendations.md` / `submission_log.jsonl` / `quota_state.yaml` / `wait_until.txt` |

`auto-kaggle` itself **never scrapes kernels, never trains, never submits.** It sequences the four skills, enforces integrity rules, manages resume / wait_until / heartbeat.

---

## Pipeline at a glance

```
                  ┌──────────────────────────────────────────────────────────┐
                  │       auto-kaggle  (orchestrator)                        │
                  └──────────────────────────────────────────────────────────┘
                                       │
   First time? ────yes──────►  Stage 0: auto-kaggle-bootstrap
                                       │   ask compute env, download data,
                                       │   parse rules, detect task type
                                       ▼
                       ┌── periodic recon ──►  Stage 1: auto-kaggle-recon
                       │                         pull top public kernels,
                       │                         distill ideas with citations
                       │                         (every N hours, configurable)
                       ▼
                  Stage 2: auto-kaggle-modeling
                     own pipeline + CV-aware training
                     + recon ideas with ablations
                       │
                       ▼
                  Stage 3: auto-kaggle-submit
                     rank by trust-adjusted CV → check quota → write recommendations.md
                     wait for user pick → actually submit → log to submission_log.jsonl
                       │
                  quota exhausted? ──yes──► write wait_until.txt, exit
                       │ no
                       └──► loop: more modeling / next recon / next submit
```

---

## Why "won't die" matters

A Kaggle run takes days to months. The agent process **will** stop for any of:

- Claude Code context fills and you `/clear`
- You close the laptop
- Network blip during `kaggle competitions submit`
- Host reboot
- Token / quota runs out

The skill is designed so all of these are recoverable:

- **All state on disk.** `.heartbeat`, `progress.jsonl`, `submission_log.jsonl`, `quota_state.yaml`, `wait_until.txt` — every file is on disk; no agent memory crosses invocations.
- **Append-only logs.** Crashes mid-write leave at worst a truncated last line; resume tolerates it.
- **Wait-until protocol.** When daily quota hits zero, the skill writes `wait_until.txt` with the next UTC midnight and exits. The supervisor sleeps until then.
- **`assets/supervisor.sh`.** A shell loop that invokes the agent headless (`claude -p ...`), survives crashes, respects STOP/PAUSE/wait_until.

Three scheduling modes, pick by how long you can keep something running:

- `manual` — you invoke `/auto-kaggle resume <slug>` whenever. Safest.
- `claude-loop` — inside Claude Code: `/loop /auto-kaggle resume <slug>`. Runs while Claude Code is open.
- `shell-supervisor` — `nohup bash auto-kaggle/assets/supervisor.sh <slug> > supervisor.log 2>&1 &` on a machine that stays up.

Stop anytime: `touch runs/<comp_slug>/STOP`.
Pause anytime: `touch runs/<comp_slug>/PAUSE` (delete to resume).
Peek at progress without invoking the agent:
```
cat runs/<comp_slug>/.heartbeat
tail -n 20 runs/<comp_slug>/progress.jsonl
cat runs/<comp_slug>/stage3_submit/recommendations.md
cat runs/<comp_slug>/stage3_submit/quota_state.yaml
```

---

## Design principles (integrity rules — `auto-kaggle/references/integrity-rules.md`)

1. **No verbatim copying** — public kernels are reference only; code is re-implemented with attribution.
2. **Every `submit -m` carries `attr:`** — naming 1–3 most influential public kernels.
3. **CV-first selection** — never rank final candidates by public LB.
4. **No LB probing** — near-duplicate submissions are blocked locally.
5. **Single account** — multi-account refused.
6. **Quota honesty** — no submitting random / placeholder predictions to burn slots.
7. **No reverse-tuning CV to match LB** — that's overfitting to the LB; private-LB suicide.
8. **Compute budget gate** — runs over budget escalate before starting.
9. **Deadline mode** — last 24h: no new architectures, only ensembling existing OOFs; final 2 submissions are user-gated.
10. **External data must be Kaggle-shared** — otherwise possible DQ.

---

## Installation

### Claude Code

```bash
mkdir -p ~/.claude/skills
cp -r auto-kaggle auto-kaggle-bootstrap auto-kaggle-recon \
      auto-kaggle-modeling auto-kaggle-submit ~/.claude/skills/
```

Or project scope: `<project>/.claude/skills/`. Then `/skills` to confirm all 5 names appear.

### Codex / OpenAI-compatible agents

Drop the same five folders into your Codex skills directory; `agents/openai.yaml` provides UI metadata.

---

## Quick start

```text
You: auto kaggle https://www.kaggle.com/competitions/playground-series-s4e5

Agent:
  → Asks: compute env / username / tier / supervisor mode
  → Stage 0: downloads data, parses rules, writes comp_profile.yaml, deadline in 27 days
  → Stage 1: pulls top 30 public kernels, distills 14 ideas with citations
  → Stage 2: runs a 5-fold LightGBM baseline, CV RMSE 0.745
  → Stage 3: writes recommendations.md with the single candidate, quota 0/5, waits for your pick
You: submit candidate 1
Agent:
  → kaggle competitions submit ..., public LB 0.748
  → Continues training new ideas...
  → Today 5/5 used: writes wait_until.txt = 2026-05-13T00:00:00Z, exits
  (Supervisor sleeps until UTC midnight, then resumes with quota reset)
```

Anytime:
```
cat runs/playground-series-s4e5/.heartbeat
cat runs/playground-series-s4e5/stage3_submit/recommendations.md
```

---

## Repository layout

```text
auto-kaggle/             # orchestrator
auto-kaggle-bootstrap/   # Stage 0
auto-kaggle-recon/       # Stage 1
auto-kaggle-modeling/    # Stage 2
auto-kaggle-submit/      # Stage 3
README.md
README.en.md
README.zh-CN.md
```

Each skill folder contains:
- `SKILL.md` — triggers + workflow
- `references/` — load-on-demand guidance
- `assets/` — helper scripts, templates, supervisor.sh
- `agents/openai.yaml` — Codex UI metadata (Claude ignores)

---

## Notes

- This is infrastructure for grinding Kaggle, **not a disclaimer**: if you violate rules, your account is still on the line.
- The final 2 submissions are always your pick; the skill ranks + recommends, never silently submits both.
- The skill **does not** modify `cv_split.yaml` on its own — once the CV scheme is set, only you change it.
- Data lives under `runs/<comp_slug>/data/` and is in `.gitignore`. Never commit it.
- Credentials (`kaggle.json`) live in `~/.kaggle/`, never under `runs/`.
