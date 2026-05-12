# Rules / Overview / Evaluation parsing

Stage 0 must extract a small set of facts from the competition's Overview, Rules, and Evaluation pages. These pages are HTML, not in the API, so they need `WebFetch` (or an equivalent scraper). Cache the raw text in `stage0_bootstrap/raw_pages/{overview,rules,evaluation}.md` so re-parsing does not re-hit Kaggle.

## What to extract

### Daily submission limit

Look in the Overview page for phrases like:
- "5 submissions per day"
- "You may submit up to N entries per day"
- "Daily Submission Limit: N"

Default: 5. Some comps allow 2, 3, or 10. Record as `comp_profile.quota.daily_limit`.

### Reset time

Look for "UTC" near the daily-limit phrase, or the comp's "What time zone is the deadline?" FAQ.

Default: `00:00 UTC`. Record as `comp_profile.quota.resets_at_utc`.

### Final selection count

Look for: "Each team may select up to N final submissions". This is almost always 2 but a few historical comps used 1 or 3. Record as `comp_profile.quota.final_selections` (default 2).

### Team size

Look in Overview's "Team Limits" or Rules' "Team Merger" section:
- "Maximum team size: N"
- "Solo entries only"

Record as `comp_profile.team.max_size`. If solo-only, also `comp_profile.team.solo_required: true`.

### Team merger freeze

Look for: "Team merger deadline: <date>". Often a week or so before final deadline. Record as `comp_profile.team.merger_freeze_utc`. Past this date, teams cannot merge — relevant if the user has been considering one.

### External data rules

Look in Rules → "External Data":
- "External data is **allowed**" → `external_data.allowed: true`
- "**Disallowed**" → `external_data.allowed: false`
- "**must be made public on Kaggle**" → `external_data.must_be_shared: true`
- "**before the deadline**" → record the deadline as `external_data.share_by_utc`

### Pre-trained models

Look in Rules → "Pre-trained Models":
- "Public pretrained weights allowed if released before <date>" → `external_data.pretrained_cutoff_utc`
- "ImageNet / generic pretraining always allowed" → `external_data.imagenet_ok: true`

This is a frequent rule violation in CV comps — record carefully.

### Code-only constraints

If `submission.code_only: true`:
- Runtime cap: "Submissions must complete within N hours"
- Max submission file size: "submission.csv ≤ N MB"
- Internet during scoring: "Internet access is disabled during scoring"
- GPU class: "Notebook must run on a T4 / P100"
- RAM cap: "Notebooks limited to N GB RAM"

Record all under `comp_profile.code_only_constraints`.

### Prize / medal tiers

Look in Overview for "Prizes" or "Tier" sections:
- "$X first place / $Y second place / ..." → `comp_profile.prizes.amounts`
- "Awards medals: Gold (top 10), Silver (top 5%), Bronze (top 10%)" → `comp_profile.medals.{gold,silver,bronze}` as a function of team count

Default medal thresholds (when comp does not state otherwise, using Kaggle's standard policy for **Featured** competitions):

| Teams | Gold | Silver | Bronze |
|---|---|---|---|
| 0–99 | top 10% | top 20% | top 40% |
| 100–249 | top 10 | top 20% | top 40% |
| 250–999 | top 10 + 0.2% × (teams − 250) | top 5% | top 10% |
| ≥ 1000 | top 10 + 0.2% × (teams − 250) | top 5% | top 10% |

Record these as `comp_profile.medals.thresholds_table` so the modeling stage can estimate where the user currently sits.

### Deadline

The comp's `deadline` field from `kaggle competitions view` is authoritative. Convert to UTC, write as `comp_profile.deadline_utc`.

Also extract from rules:
- Entry deadline (last date to join/team-merge — `comp_profile.entry_deadline_utc`)
- Final submission deadline (same as `deadline` usually)
- Rules acceptance deadline (same as deadline for most comps)

### Anti-cheating clauses

Scan the Rules page for clauses about:
- "Hand-labeling" — usually forbidden for image / segmentation comps
- "Probing the leaderboard" — universally forbidden
- "Reverse engineering the test set" — forbidden
- "Account sharing" — forbidden

Record verbatim in `rules_summary.md` under a "Cheating constraints" section. These map to integrity rules and the agent must respect them at every later step.

## How to call `WebFetch`

The bootstrap skill writes the URL it wants fetched + the parsing question:

```
WebFetch(
  url="https://www.kaggle.com/competitions/<slug>/rules",
  prompt="Extract: daily submission limit, team size limit, external data rules,
          pre-trained model rules, code-only runtime cap, any anti-cheating clauses.
          Return as a YAML block matching this schema: <inline schema>"
)
```

Cache the raw response to `stage0_bootstrap/raw_pages/rules.md` before parsing, so re-runs do not re-hit Kaggle.

Login-walled content: most Kaggle pages are public, but discussion threads and some leaderboards require login. If `WebFetch` returns a login redirect, escalate — the user may need to provide a logged-in HTML dump.

## `rules_summary.md` shape

```markdown
# <Comp title>

**URL:** <url>
**Deadline (UTC):** <ts>
**Category / Tier:** <Featured / Research / Playground Series>

## Submission
- Daily limit: <N> (resets 00:00 UTC)
- Final selections: <N>
- Format: <csv | code-only>
- Size cap: <N> MB
- Code-only constraints: <N>h runtime, <N> GB RAM, internet <on|off>, GPU <class>

## Teams
- Max size: <N>
- Solo required: <yes|no>
- Merger freeze: <ts>

## External data
- Allowed: <yes|no>
- Must be shared on Kaggle: <yes|no>
- Pretrained cutoff: <ts or "imagenet ok">

## Cheating clauses (verbatim)
> "Hand-labeling is not permitted ..."
> "Probing the public leaderboard ..."

## Medals (estimated)
- Gold: top <N> / X teams
- Silver: top <N>
- Bronze: top <N>

## Notes
- <anything weird or worth flagging>
```

Keep it under 100 lines. Every later stage reads this.
