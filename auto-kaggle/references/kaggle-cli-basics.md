# Kaggle CLI basics (what the skill actually uses)

Reference for the Kaggle API commands the four stage skills invoke. Stick to these — they are stable, well-documented, and respect Kaggle's rate limits. Do not screen-scrape what the API already provides.

## Setup (one-time per machine)

```bash
pip install --upgrade kaggle
mkdir -p ~/.kaggle
# Place kaggle.json (downloaded from Kaggle account settings → "Create New API Token") here:
chmod 600 ~/.kaggle/kaggle.json
```

The skill **never** reads `kaggle.json`. It assumes the `kaggle` CLI is on PATH and authenticated.

## Commands by stage

### Stage 0 (bootstrap)

```bash
# Confirm comp exists and get high-level metadata
kaggle competitions list -s "<slug>"

# Read the comp's metadata (returns JSON-ish)
kaggle competitions view <slug>

# Download the data (huge, may take long; tracked in progress.jsonl as data_downloaded)
kaggle competitions download -c <slug> -p runs/<slug>/data/raw/

# Unzip (kaggle downloads .zip)
cd runs/<slug>/data/raw && unzip -o "*.zip" && rm *.zip
```

### Stage 1 (recon)

```bash
# List top public kernels by vote (highest first)
kaggle kernels list -c <slug> --sort-by voteCount --page-size 50

# List by recency
kaggle kernels list -c <slug> --sort-by dateCreated --page-size 50

# Pull a kernel's source + metadata to a folder
kaggle kernels pull <author>/<kernel-slug> -p runs/<slug>/stage1_recon/kernels/<author>__<kernel-slug>/ -m

# Search dataset listings the kernels depend on
kaggle datasets list -s "<search-term>"
```

`kaggle kernels list` returns a table; pipe to `--csv` for parsing:

```bash
kaggle kernels list -c <slug> --sort-by voteCount --page-size 50 --csv > kernels_top_votes.csv
```

### Stage 3 (submit)

```bash
# Submit a CSV
kaggle competitions submit -c <slug> -f <path-to-csv> -m "<message>"

# List your past submissions for this comp (authoritative quota source)
kaggle competitions submissions -c <slug>

# Current public LB (use sparingly — informational only, not for selection)
kaggle competitions leaderboard -c <slug> --show
```

`kaggle competitions submissions -c <slug>` is the **source of truth** for "how many submissions did I make today." The skill reconciles `quota_state.yaml` against this on every resume.

## What the CLI does **not** give you (and how to handle it)

| Need | API gives | Workaround |
|---|---|---|
| Today's remaining submissions | No direct field | Compute from `kaggle competitions submissions -c <slug>` filtered by `date == today UTC` |
| Discussion threads | No | Use `WebFetch` or a scraper — see `external-tools.md` §1 |
| Kernel output (rendered notebook) | Only source via `kaggle kernels pull` | Scrape the notebook URL if output is needed |
| Code-only submission status | `kaggle competitions submissions` shows it after build | Poll on a backoff after submitting |
| Team membership | Not in API | User manages in `run.yaml.team` manually |

## Rate limits

Kaggle's CLI is rate-limited (exact numbers undocumented; treat as ~60 requests/min). The skill must:

- Cache `kaggle competitions view <slug>` output in `stage0_bootstrap/raw_comp_view.json` and re-read for up to 24h.
- Pull at most 50 kernels per recon cycle, spaced 1s apart.
- Use `kaggle competitions submissions` at most once per submit attempt and once per resume.

If you see HTTP 429 or "Too Many Requests," back off (60s, then 5min, then 30min) and log `progress.jsonl` event `rate_limited`.

## Authentication errors

If a command returns `401 Unauthorized` or `403 Forbidden`:

- Do **not** retry blindly — the user may have not yet accepted the competition rules on the website (a common 403 cause).
- Escalate: write to `recommendations.md` "Auth failed for `<command>`. Common causes: (a) `kaggle.json` missing or wrong, (b) competition rules not accepted on website, (c) account banned. Please verify."
- Exit. Do not consume retries.

## Environment variables the skill respects

```
KAGGLE_USERNAME, KAGGLE_KEY              # auth (alternative to kaggle.json)
KAGGLE_CONFIG_DIR                        # override ~/.kaggle/
AUTO_KAGGLE_PROXY                        # if set, prepend `https_proxy=...` to kaggle commands
```
