# Ranking rubric for `recommendations.md`

The core artifact Stage 3 produces. This is what the user reads before picking. Every entry must be honest, comparable, and traceable — never optimize the ranking for "looks impressive."

## Trust-adjusted CV

```
direction = comp_profile.metric.direction        # "maximize" | "minimize"
α         = run.yaml.trust_alpha                  # default 1.0

if direction == "maximize":
    trust_adjusted = cv_score - α * cv_std
else:
    trust_adjusted = cv_score + α * cv_std
```

The penalty `α * cv_std` is what makes "high CV mean, high variance" lose to "slightly lower CV mean, low variance." Without it, a run that fluked into a high fold and an average fold elsewhere wins, and the user submits something that won't generalize.

`α` defaults to 1.0. Increase to 1.5 if the CV-LB gap has been > 1 std on the last 3 submissions (the leaderboard is noisier than CV — punish variance harder). Decrease to 0.5 if the gap has been < 0.5 std for the last 5 submissions (CV is well-calibrated; you can afford to trust the variance estimate less).

## The two reserved slots: RECOMMENDED and SAFE

Out of all completed, trustworthy, non-near-duplicate runs:

- **RECOMMENDED:** highest `trust_adjusted`. The model that should win on average. If a single model is in the top by both `trust_adjusted` and raw `cv_score`, this is clearly it.
- **SAFE:** lowest `cv_std` among the top 5 by `trust_adjusted`. The model whose CV you trust most as a predictor of private LB. May coincide with RECOMMENDED — in that case, mark `RECOMMENDED also SAFE` and skip the second slot to a different candidate (the second-lowest cv_std).

Why both? Because for the 2 final submissions you want one "central" and one "diverse." For day-to-day submissions you might want either, depending on how much information you've gathered about the LB calibration.

## Candidate filters

A run goes into the ranked list iff:

- `status == "completed"` (not failed, not terminated_overbudget)
- `attribution.md` Citations section is non-empty OR has `+own` (Rule 2)
- `cv_std` is set (otherwise we can't compute trust_adjusted)
- Not flagged as `near_duplicate` of any run already in `submission_log.jsonl`

A run is **excluded from RECOMMENDED / SAFE slots** but still shown in the list iff:

- `cv_std / |cv_score| > 0.5` (`untrustworthy`)
- `near_duplicate: true` (still shown with a ⚠ marker for transparency)

A run is **dropped entirely** iff:

- The run dir is missing critical files (`test_preds.csv` or `attribution.md`).

## Near-duplicate detection

Loaded `near_duplicate.py` does:

```python
def is_near_duplicate(candidate_preds, prior_preds_list, threshold=1e-6):
    for prior in prior_preds_list:
        if candidate_preds.shape != prior.shape:
            continue
        mae = float(np.mean(np.abs(candidate_preds - prior)))
        if mae < threshold:
            return True, mae, prior_run_id
    return False, None, None
```

Threshold 1e-6 by default. Why: legitimate ensembles always change predictions by orders of magnitude more. A pair within 1e-6 is either an identical model with different seed (recompute the seed_avg ensemble instead of submitting both) or an attempt at LB probing.

Override: pass `--allow-near-duplicate <run_id> --reason "<text>"` to the submit command. The reason is written into `recommendations.md` next to that candidate for the user to read.

## "Other ranked candidates" table format

Below RECOMMENDED and SAFE, show all remaining candidates in a table:

| Rank | run_id | CV | std | Trust-adj | Public LB | Status | Notes |
|---|---|---|---|---|---|---|---|
| 3 | … | … | … | … | … | completed | … |
| 4 | … | … | … | … | … | completed | (untrustworthy: std/cv = 0.6) |
| ⚠ | … | … | … | … | … | completed | near_duplicate of #1 — excluded |
| - | … | failed (template error) |

Rank `⚠` for excluded-but-shown rows; rank `-` for failed rows (educational; user sometimes wants to know which ideas blew up).

## How to phrase reasoning

Each top slot gets a "Why this" sentence. Templates:

- "Best trust-adjusted CV. CV–LB gap is within 1 std → CV is a good predictor of LB."
- "Smallest variance among the top 5 — safest bet if the higher-CV blend overfits."
- "Diverse ideas — three independent models contributing, low risk of correlated errors."
- "Highest raw CV; variance is moderate but CV is far enough ahead to absorb the penalty."

Avoid: empty praise, vague claims ("looks great"), "this should win" (private LB has no should).

## CV-LB calibration health line

At the top of `recommendations.md`, after the quota line, include one of:

- `CV-LB calibration: healthy` — last 3 submissions have gaps < 1 × cv_std
- `CV-LB calibration: mild` — last 3 submissions have gaps 1–2 × cv_std
- `CV-LB calibration: suspect` — gaps > 2 × cv_std on any of last 3 (escalate per `escalation-policy.md` if not already)

The user reads this and decides whether to weight CV or LB more in their pick.

## Day-to-day vs deadline-mode ranking

Day-to-day (not last 24h):

- Rank by `trust_adjusted` (standard).
- Top slot RECOMMENDED.
- Second slot SAFE.
- The "burner" submission strategy: submit RECOMMENDED most days; submit SAFE when LB calibration is suspect or you want a second opinion.

Deadline mode (last 24h):

- Same ranking, but the top 2 are renamed `FINAL CANDIDATE 1 (SAFE)` and `FINAL CANDIDATE 2 (AMBITIOUS)`.
- The skill **does not auto-submit** these; user must `--final-submission-confirm`.
- The user picks both at deadline-time — and that pair is the final-2 for private-LB scoring.

In deadline mode the rubric prefers diversity for the two final picks:

```
final_picks = (
    (most trustworthy single model — lowest cv_std among top 3 by trust_adjusted),
    (most diverse high-quality candidate — highest trust_adjusted among candidates whose
     test_preds disagreement with the SAFE pick is > median pairwise disagreement of top-10)
)
```

This is what `final-selection.md` proposes; the user can swap either.

## Things the rubric explicitly avoids

- Ranking by raw public LB. Rule 3.
- Ensembling predictions automatically without an explicit ensemble run. Stage 2 produces ensembles as their own runs; Stage 3 only picks among completed runs.
- Hiding "near duplicate" runs. They appear in the list with a ⚠ so the user can ask "why are these two so similar."
- "Computing the optimal pair" — that's overfitting our own choices to a tiny sample of LB feedback. Two slots with simple criteria are robust.
