# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A library of **five portable skill packages** for Claude Code and Codex-style agents that together drive multi-day Kaggle medal hunting from a single competition URL. It is *not* a runnable application — there is no top-level build, no test suite, no `requirements.txt`. The deliverables are markdown + YAML + Python helper scripts + a shell supervisor that get copied into `~/.claude/skills/` (or `<project>/.claude/skills/`) and executed by an agent at runtime.

Implication for editing: the "code" here is the prompts, workflow contracts, and a handful of Python helpers. Changes are validated by reading them and tracing the contract, not by running them inside this checkout. Do not try to `pip install` or `kaggle competitions ...` from here to "verify" a change — real runs happen on the user's machine.

## The five skills and how they compose

```
auto-kaggle              ← orchestrator (Stage 0..3 driver, integrity gate, resume protocol)
  ├─ auto-kaggle-bootstrap   ← Stage 0: ask compute env, download data, parse rules, detect task type
  ├─ auto-kaggle-recon       ← Stage 1: periodic top-public-kernel pulls, distill ideas with attribution
  ├─ auto-kaggle-modeling    ← Stage 2: own CV-aware pipeline, integrate recon ideas with ablations
  └─ auto-kaggle-submit      ← Stage 3: rank candidates, track quota, recommend, submit on user pick
```

Each skill folder has the same shape:

- `SKILL.md` — frontmatter (`name`, `description`) + workflow. The description triggers the skill in Claude Code / Codex; keep it under 1024 chars.
- `references/*.md` — load-on-demand reference docs cited from `SKILL.md`'s "When to load which reference" table. Skills should NOT load all references upfront.
- `assets/` — concrete artifacts: helper scripts (`kaggle_helpers.py`, `bootstrap.py`, `supervisor.sh`), modeling templates (planned), and any LaTeX-free notebook templates for code-only submission packaging.
- `agents/openai.yaml` — Codex-side UI metadata only. Claude Code ignores it. When changing skill metadata, keep `SKILL.md` frontmatter and `openai.yaml` in sync.

The orchestrator never does the modeling itself; it sequences the four sub-skills and enforces the contract between them.

## State contract — do not invent paths

All inter-stage state lives under `runs/<comp_slug>/` (where `<comp_slug>` is the **exact** slug from the Kaggle URL — never invented). The full file schema is defined in `auto-kaggle/references/state-contract.md` — treat that file as authoritative. When editing any skill that reads or writes stage artifacts, cross-check it against `state-contract.md` so the contract stays consistent across all five skills.

Files every stage reads first:

- `runs/<comp_slug>/run.yaml` — comp slug, compute env, target tier, deadline, supervisor mode
- `runs/<comp_slug>/.heartbeat` — current stage/substep/ts_utc/pid (user can `cat` to peek)
- `runs/<comp_slug>/progress.jsonl` — append-only micro-step log; resume reads its tail
- `runs/<comp_slug>/stage{N}_*/hand_off.md` — one-paragraph hand-off from stage N to stage N+1
- `runs/<comp_slug>/stage0_bootstrap/comp_profile.yaml` — task type / metric / quota / deadline

`runs/` is gitignored — never commit experimental output back into this repo.

## The long-running protocol overrides everything

`auto-kaggle/references/long-running-protocol.md` defines how the skill survives across invocations:

- All state on disk; no agent memory crosses invocations.
- Idempotent micro-steps; re-running re-reads disk and continues.
- Append-only logs (`progress.jsonl`, `submission_log.jsonl`); never rewritten.
- Atomic file writes via `.tmp` + rename.
- Resume by default; `--restart` required for a fresh start.
- `wait_until.txt` is the sleep-until-quota-resets protocol.
- `STOP` / `PAUSE` sentinels in `runs/<comp_slug>/` are honored at the top of every micro-step.
- `assets/supervisor.sh` keeps the run alive across crashes from outside Claude Code.

When editing any skill: every new code path must say how it satisfies (or escalates under) the long-running protocol. If a new operation is not idempotent, that is a bug.

## The integrity rules override everything else

`auto-kaggle/references/integrity-rules.md` defines the 10 non-negotiable rules (no verbatim copying, attribution mandatory, CV-first selection, no LB probing, single account, quota honesty, honest CV, compute budget, deadline mode, external data must be Kaggle-shared). When editing any skill:

- Do not loosen these rules to make a workflow easier.
- Final-submission selection is user-picked, always. The skill ranks and recommends — never auto-picks the 2 final submissions.
- The orchestrator's "stop and ask the human" checkpoints listed in `escalation-policy.md` are part of the contract — preserve them when touching the orchestration loop.

## Conventions when editing skills

- Keep `SKILL.md` frontmatter `description` within Claude Code's 1024-character limit and unambiguous about when to trigger.
- Examples in skills use absolute dates and absolute UTC timestamps. Never write "today" or "yesterday" — those rot.
- The repo intentionally has both English (`README.en.md`) and Chinese (`README.zh-CN.md`) READMEs; if you change one substantively, mirror the change in the other.
- Helper scripts (`bootstrap.py`, `kaggle_helpers.py`, `supervisor.sh`) follow `state-contract.md` for input/output paths. Do not invent new paths in helpers.
- `kaggle.json` and any credential file are out of scope for this repo. The skill reads CLI auth from `~/.kaggle/` only.
- `.gitignore` includes `runs/`, `kaggle.json`, `.kaggle/`, `__pycache__/`, `._*`, large data extensions. Verify before committing.

## Working in this environment

This machine is shared and resource-constrained — it is for *editing skills and committing*, not for executing them. Do not attempt to invoke a skill end-to-end from this checkout to test it: actual runs (Kaggle API calls, training, submissions) belong on the user's local / cloud machine. When the user wants to validate a change, list the commands they should run there rather than running them here.
