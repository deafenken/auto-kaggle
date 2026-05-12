# Discussion mining

An optional Stage 1 sub-step that scrapes the competition's Discussion tab for high-signal posts (rule clarifications, leak rumors, baseline announcements, mid-comp shifts). Surfaces ideas and warnings that kernel scraping misses.

This is **opt-in** — set `run.yaml.recon_discussion.enabled: true` to turn on. Default off because it requires `WebFetch` and the signal-to-noise is comp-dependent (some comps have stellar discussion, some have noise).

## Why it matters

Public kernels show "what people have done." Discussions show "what people are *talking* about doing, what's broken, and what they suspect the LB is hiding."

Specific failure modes the discussion catches but kernels do not:
- **Data leaks** discovered by participants and posted (sometimes patched by organizers; sometimes not).
- **Rule clarifications** — "Hand-labeling clarification: it IS allowed if …" — these change strategy.
- **Mid-comp data swaps** — rare but happens; organizers re-upload corrected data.
- **Score plateau patterns** — multiple competitors describing the same wall ("nobody is breaking 0.85") signals an architectural ceiling.
- **Stake-up rumors** — "this comp will shake-up because the public LB uses only X% of test."

## What we DON'T scrape

- Personal opinions / off-topic threads.
- Vendetta / community drama.
- "How do I install pytorch" support questions.
- Threads with < 5 votes and < 3 replies.

## Trigger and throttle

Lives behind the same throttle window as kernel recon (`run.yaml.recon_interval_hours`). The orchestrator calls `recon_discussion.py` AFTER `recon.py` completes successfully and only if discussion mining is enabled.

`last_discussion_at` is tracked separately from `last_recon_at` so the two cadences can drift apart if needed.

## Inputs

- `runs/<comp_slug>/stage0_bootstrap/comp_profile.yaml`
- The Kaggle competition's Discussion page URL: `https://www.kaggle.com/competitions/<slug>/discussion`
- `runs/<comp_slug>/stage1_recon/last_discussion_at` (if exists)
- `runs/<comp_slug>/stage1_recon/discussion_index.json` (if exists; for delta)

## Outputs

```
runs/<comp_slug>/stage1_recon/
├── discussion_index.json     # one entry per scraped thread
├── discussions/<thread_id>/  # raw markdown of each thread, for reference
├── last_discussion_at        # ISO-8601 UTC of last scrape
└── (additions to existing files:)
    ├── ideas_pool.md         # new entries with `discussion:` citations
    └── citations.bib         # discussion entries (URL-based, not kernel-based)
```

The `discussion_index.json` schema (per thread):

```json
{
  "thread_id": "452317",
  "url": "https://www.kaggle.com/competitions/<slug>/discussion/452317",
  "title": "Suspected leak in test_id 12345",
  "author": "jdoe123",
  "posted_utc": "2026-05-11T15:20:00Z",
  "votes": 47,
  "replies": 23,
  "last_pulled_utc": "2026-05-12T10:00:00Z",
  "category": "leak | rule-clarification | technique | baseline | shake-up | other",
  "high_signal": true,
  "summary": "Author found that test_id 12345 ...",
  "distilled_at_utc": "2026-05-12T10:32:00Z"
}
```

## Scrape pattern (via WebFetch)

The Discussion page is not in the Kaggle API. Use `WebFetch` (or an equivalent):

```
WebFetch(
  url="https://www.kaggle.com/competitions/<slug>/discussion?sort=hotness",
  prompt="""Extract all visible threads. Return a JSON array of objects with:
  - thread_id (the numeric ID in the URL after /discussion/)
  - title
  - author
  - posted_utc (ISO-8601)
  - votes (integer)
  - replies (integer)
  - first 200 chars of the post body
Skip pinned official threads from kaggle staff. Return [] if the page is empty.
"""
)
```

Then for each thread above a vote threshold (`votes >= 10` by default), fetch the full thread body:

```
WebFetch(
  url="https://www.kaggle.com/competitions/<slug>/discussion/<thread_id>",
  prompt="""Return the OP body and the top 5 replies (by votes) as markdown.
Strip emojis, image-only posts, and code blocks longer than 50 lines.
"""
)
```

Save the markdown to `discussions/<thread_id>/post.md`. Save the metadata to `discussion_index.json`.

## Categorization (agent does this, not the script)

After scraping, the agent reads the markdown for each new thread and classifies into one of:

- **`leak`** — claims a data leak (verify carefully, often false alarms but the real ones are gold)
- **`rule-clarification`** — organizer or hosted-staff post clarifying a rule
- **`technique`** — describes a specific approach (CV scheme, augmentation, post-processing)
- **`baseline`** — a starter or strong baseline announcement
- **`shake-up`** — meta-discussion of public-vs-private LB calibration
- **`other`** — fluff or out-of-scope

`high_signal: true` if any of:
- Author is a known top-tier Kaggler (GM with ≥3 golds — agent infers from author profile if needed)
- Category is `leak` or `rule-clarification`
- Votes / replies > 20

## Where ideas go

For each high-signal `technique`-category thread:

1. Add an entry to `ideas_pool.md` with category mapped from the discussion's substance (`cv` / `feature` / `model` / etc.).
2. Citation format in `ideas_pool.md`: `discussion:<thread_id>` instead of `<author>/<kernel-slug>`. The submission's `attr:` token uses the discussion URL.
3. Add an entry to `citations.bib`:
   ```bibtex
   @misc{discussion-452317,
     author = {jdoe123},
     title  = {Suspected leak in test_id 12345},
     year   = {2026},
     url    = {https://www.kaggle.com/competitions/<slug>/discussion/452317},
     note   = {Kaggle discussion thread, pulled 2026-05-12, votes 47},
     key    = {discussion-452317}
   }
   ```

For each `rule-clarification` or `leak` thread, regardless of `high_signal`:

- Append a `⚠` line to the next Stage 1 → Stage 2 `hand_off.md` calling it out.
- If `category == "leak"` and the agent assesses it as plausible (not folklore), append a hard escalation to `runs/<comp_slug>/stage3_submit/recommendations.md` — the user must decide whether to use, ignore, or report it.

## Integrity considerations

- **Don't republish private content.** If a discussion thread contains a leaked test set sample or copyrighted data, do NOT copy it into our artifacts. Reference the thread URL only.
- **Don't act on rumors.** A thread claiming a leak is not proof. Mark as `claim` until verified.
- **Don't ignore organizer posts.** Posts by Kaggle staff (Kaggle's official account or comp host) are authoritative for rule clarifications. They override our cached `rules_summary.md` — bootstrap should re-run if a thread changes rules.
- **Rate limits.** WebFetch is slower than the kaggle CLI; cap discussion pulls at 30 threads per cycle.

## When discussion mining is NOT worth turning on

- Very new competitions (< 1 week old, < 50 teams) — discussion is empty.
- Comps in a domain you know well — you have priors and don't need crowd signal.
- Restricted time (< 3 days to deadline) — focus on training, not reading.

## When discussion mining IS worth turning on

- Comps with active forums (≥ 500 teams, ≥ 50 discussion threads) — there's real signal.
- Comps with rumored leaks or shake-ups — discussion is where these surface.
- Comps where you're behind the medal cutoff and need an unconventional edge.
