# Final selection (deadline mode)

The single most important decision in the entire run. The 2 submissions you pick at deadline are the ones that count for private LB and medals.

The skill never picks autonomously here. Rule 3 + Rule 9 — these are user-gated. The skill **proposes** and **explains**; the user **decides**.

## When deadline mode triggers

`now > comp_profile.deadline_utc - 24h`. The orchestrator checks at the start of every cycle and sets `deadline_mode: true` in the heartbeat.

In deadline mode:

- Stage 1 (recon) still runs but stops adding new ideas — only refreshes the index.
- Stage 2 (modeling) refuses new architectures; only ensembles existing OOFs.
- Stage 3 (submit) refuses to submit without `--final-submission-confirm`.
- A new file `final_selection.md` appears in `stage3_submit/`.

## How the two slots are chosen (proposed, not enforced)

Rule of thumb: **pick two that disagree on something concrete**, not two near-duplicates.

### Slot 1 — SAFE

```
candidates = top 5 by trust_adjusted_cv
slot1 = candidate with smallest cv_std among candidates
        AND cv_score within 1 std of the best in candidates
```

Logic: smallest variance among the high-CV pool. If private LB is close to public LB, this loses you a few places (vs. the absolute best CV) but never collapses. The "within 1 std of the best" gates that we're not picking a low-variance mediocre run.

### Slot 2 — AMBITIOUS

```
candidates = top 10 by trust_adjusted_cv, excluding slot1
slot2 = candidate that maximizes
            trust_adjusted_cv(c)
          - β * pairwise_correlation(c, slot1)
```

Where `pairwise_correlation` is the Pearson correlation between candidate's `test_preds.csv` and slot1's `test_preds.csv`, and `β` is the diversity premium (default 0.5). Higher `β` favors picking something that disagrees with slot1, which gives shake-up diversification.

If the candidate's correlation with slot1 is > 0.99, drop it from the candidate set — it's too similar to count as diversification.

Logic: if private LB shifts in a direction we can't predict, having two highly-correlated picks means both lose. Having one safe + one diverse gives us a chance of catching the shift.

## What `final_selection.md` contains

```markdown
# Final selection — <slug>
_Deadline: <ts> (in HH:MM). Created at <ts>._

## Slot 1 — SAFE — <run_id>
- **CV (<metric>):** <score> ± <std>
- **Public LB (last):** <score> · gap <delta>
- **Why proposed:**
    Lowest cv_std (<std>) among the top 5 by trust-adjusted CV. CV–LB
    gap of <delta> ≤ 1 std, so we trust private LB tracks CV.
- **Risk profile:** If CV is well-calibrated, this places competitively.
  If CV is over-optimistic, this still ranks higher than ablation-level CV runs.
- **Ideas in this run:** <list>
- **Attribution:** attr: <list>
- **Predictions sample (first 3):**
    id_1: 0.7421
    id_2: 0.5183
    id_3: 0.9012

## Slot 2 — AMBITIOUS — <run_id>
- **CV (<metric>):** <score> ± <std>
- **Public LB (last):** <score> · gap <delta>
- **Disagreement with slot 1:** Pearson 0.81, mean abs diff 0.12 on test
- **Why proposed:**
    Highest trust-adjusted CV among runs that disagree substantially with
    slot 1. <One-paragraph specific reasoning for what makes this risky-but-worth-it>.
- **Risk profile:** Higher variance; private LB could be 0.01 better or 0.01
  worse than slot 1's. If the comp's hidden test set has the characteristics
  our blend was tuned for, this wins.
- **Ideas in this run:** <list>
- **Attribution:** attr: <list>

## What you should do now
1. Read both slots above.
2. Optionally inspect:
   - `runs/<slug>/stage2_modeling/runs/<slot1>/attribution.md`
   - `runs/<slug>/stage2_modeling/runs/<slot2>/attribution.md`
   - `runs/<slug>/stage2_modeling/leaderboard.csv`
3. If you accept both proposals, run:
       auto-kaggle-submit <slug> --final-submission-confirm --slot1 <run_id> --slot2 <run_id>
4. If you want different picks, edit `final_selection.md` to replace the
   `Slot 1` / `Slot 2` blocks with the runs you want, then run the same
   command.
5. After the skill submits, go to the Kaggle website and **manually select
   these two submissions** as your final (Kaggle has no API for this).
```

## What the skill does NOT do

- Auto-submit the final 2 without `--final-submission-confirm`.
- Click "Select for Final" on the Kaggle website. The user does this manually.
- Pick different slots than the user requested. If the user says "use these two specific run_ids," the skill submits exactly those.

## Edge cases

### Tied trust_adjusted_cv across many runs

If the top 5 are within 0.5 × cv_std of each other, the safety-via-trust signal is too weak. Switch to:

- Slot 1 = the one with the most consistent CV–LB gap on past submissions.
- Slot 2 = the most diverse from slot 1.

Document this in `final_selection.md` so the user sees why.

### Only one decent run

If there is only one completed, trustworthy run, slot 2 should be one of:

- The same run with a different seed (a multi-seed average computed on the fly as a new ensemble run before final selection)
- A blend of the top 3 runs (an ensemble run, even if individual runs are mediocre)
- The best non-trustworthy run (with a loud "this is gambling because we have no diversification" warning)

In any case, explain the situation clearly. Never pretend two near-identical runs are diverse.

### CV–LB gap has been consistently large

If the gap was 2+ × cv_std on the last 3 submissions, CV is not predicting LB well. In this case:

- Slot 1 = best public LB (the only signal we have that maps to private LB)
- Slot 2 = best CV (still our best estimate of private LB structure)

Flag this with a loud `⚠ CV-LB calibration suspect — slot 1 chosen by LB, not CV` in the doc. Rule 3's spirit (CV-first) bends here because CV is demonstrably unreliable; the user should know.

### Team competition

If `run.yaml.team` lists multiple members, the team gets `final_selections` (per-team count, usually 2) total picks across all teammates. The user is responsible for coordinating with teammates on which 2 to pick. The skill produces `final_selection.md` with proposals but explicitly says "coordinate with teammates before confirming."

## After submission

When the user confirms and the skill submits both:

1. Log both in `submission_log.jsonl` with `is_final_candidate: true`.
2. Update `final_selection.md` with `final_picked_at_utc` per slot.
3. Print: "Submitted both. Now go to https://www.kaggle.com/competitions/<slug>/submissions and click 'Select for Final' next to each of these two submissions. The skill cannot do this for you."
4. Append `progress.jsonl` event `final_selections_submitted`.

The skill then idles. Until the user signals (a STOP file appears, or deadline passes), it does nothing further. Deadline-passed → supervisor exits.
