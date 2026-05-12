# Compute environment options

The bootstrap stage asks the user to pick one of four environments. The choice propagates into every subsequent stage's runtime budget, batch size defaults, and training template selection.

## The four options

### 1. `kaggle-notebook`

Kaggle's hosted notebooks (free tier).

| Property | Value |
|---|---|
| GPU | 2× NVIDIA T4 (15 GB each) **or** 1× P100 (16 GB) |
| GPU quota | 30 hours / week, hard-resets weekly |
| Wallclock per kernel | 9 hours (commits) / 12 hours (uncommitted) |
| RAM | 30 GB |
| Disk | 20 GB ephemeral, 75 GB extra `/kaggle/working` if you stay within the kernel |
| Internet | On in editor, **OFF** in scored submission run for code-only comps |
| When mandatory | Code-only competitions — `comp_profile.submission.code_only: true` forces this |
| Pros | Free, fast iteration, no infra setup, fits Kaggle's submission flow natively |
| Cons | Tight wallclock, no persistence between kernels except via Kaggle Datasets |

**Capability defaults set by bootstrap:**

```yaml
specs:
  gpu_model: T4    # or P100; user picks per kernel via the kernel UI
  vram_gb: 15
  cpu_cores: 4
  ram_gb: 30
  internet: true
constraints:
  max_wallclock_per_run_hours: 8.5   # buffer below the 9h hard limit
  parallel_runs: 1
  gpu_hours_per_week: 30
```

### 2. `local-gpu`

User's own machine.

| Property | Value |
|---|---|
| GPU | Whatever the user has (ask: model + VRAM) |
| GPU quota | Unlimited (user-paid power) |
| Wallclock per kernel | Unlimited |
| RAM | User-supplied |
| Internet | Yes |
| When good | Mid- to long competitions where you'll train many variants; comps with large data |
| Pros | No quota, fastest iteration if you have the hardware |
| Cons | You own infra problems; cooling, power, drivers |

**Ask the user:**

```
Local GPU details:
  - GPU model (e.g. RTX 3090):
  - VRAM in GB:
  - System RAM in GB:
  - CPU cores:
  - Free disk in GB (for runs/<slug>/data/):
```

Validate: VRAM ≥ 8 GB for CV/NLP, ≥ 4 GB for tabular. If lower, downgrade to cpu-only with a warning.

### 3. `cloud-gpu`

User rents a GPU somewhere: Colab Pro/Pro+, Lambda Labs, Vast.ai, RunPod, AWS, etc.

The skill does not manage the instance. The user is responsible for keeping it alive.

**Ask the user:**

```
Cloud GPU details:
  - Provider (Colab Pro / Lambda / Vast / RunPod / AWS / other):
  - GPU model:
  - VRAM in GB:
  - Hours/day you keep it on:
  - SSH or notebook access?
```

The supervisor mode `shell-supervisor` works here if the user can keep an SSH session alive (use `tmux` / `screen`).

### 4. `cpu-only`

No GPU available, or VRAM too small for the task type.

| Use case | Reasonable |
|---|---|
| Small tabular comp (< 1M rows) | Yes — LightGBM/XGBoost on CPU is competitive |
| Tabular comp > 5M rows | Marginal — long training times |
| CV comp | No — image models are GPU-bound |
| NLP comp | Only for traditional ML (tf-idf + LR) — transformer fine-tuning needs GPU |
| Time-series forecasting | Yes if you're using statistical or tree models |

**Capability defaults:**

```yaml
specs:
  gpu_model: null
  vram_gb: 0
  cpu_cores: <user-supplied>
  ram_gb: <user-supplied>
  internet: true
constraints:
  max_wallclock_per_run_hours: 4
  parallel_runs: 1
```

Stage 2 modeling will:
- Refuse to start any model whose preferred backend is `cuda`.
- Default to `LightGBM` / `CatBoost` (`device='cpu'`) for tabular tasks.
- Escalate immediately for CV / NLP-transformer tasks: "this comp's task type is incompatible with cpu-only. Switch compute env, or accept a non-medal floor."

## Compute-env-driven defaults the modeling stage uses

| Env | Default fold count | Default seed count | Default ensemble size |
|---|---|---|---|
| kaggle-notebook | 5 | 1 (multi-seed via separate runs) | 3 (blend) |
| local-gpu | 5–10 (task-dependent) | 3 | 5 |
| cloud-gpu | 5–10 | 3 | 5 |
| cpu-only | 5 | 1 | 3 |

These are starting points; modeling stage tunes them based on `compute_env.yaml.constraints.max_wallclock_per_run_hours`.

## Parsing the user's answer

The bootstrap skill expects one of the four exact strings: `kaggle-notebook`, `local-gpu`, `cloud-gpu`, `cpu-only`. If the user says something like "I have a 3090", interpret as `local-gpu` and fill the specs. If they say "Colab", interpret as `cloud-gpu` and ask the followup. If they say nothing about a GPU and they're on macOS (no NVIDIA), default to `cpu-only` and confirm.

Never silently pick `kaggle-notebook` — that's a serious constraint (no internet at scoring, 9h cap) and the user must opt in.
