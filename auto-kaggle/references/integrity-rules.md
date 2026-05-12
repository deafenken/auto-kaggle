# Integrity rules — non-negotiable

These rules override convenience, speed, and "what would score higher." A skill that finds itself about to violate one of these must **stop and escalate** (see `escalation-policy.md`), not work around it.

Why these exist: Kaggle has both a written ToS and an unwritten community-policed culture. Violations get you (a) DQ'd from the competition, (b) banned from Kaggle, (c) reputationally torched in a way that follows you through grad school applications and job interviews. A few extra LB points are not worth any of that.

---

## 1. No verbatim copying from public kernels

Public Kaggle kernels are licensed (usually Apache-2.0 or CC BY-SA 4.0, sometimes proprietary). You may **read and learn from** them. You may **not** lift code blocks larger than a function signature into your pipeline without attribution.

**Operational rule:** Every file under `stage2_modeling/` that contains code derived from a public kernel must have, at the top, a `# Derived from: <kernel-author>/<kernel-slug> (<license>) — see citations.bib` comment, and the kernel must appear in `stage1_recon/citations.bib`.

If you need a technique from a kernel, **re-implement it from scratch** based on the idea, not the code. The re-implementation goes in `stage2_modeling/pipeline.py` with an attribution comment naming the kernel that inspired it.

## 2. Attribution in every submission message

Every call to `kaggle competitions submit` must use a `-m "..."` message that lists the top 1–3 public kernels that influenced this submission, in the form `attr: <author>/<kernel-slug>`. Skipping attribution is treated as a covert claim of originality.

Example: `-m "lgbm 5fold + augment_v2 + tabnet stack | attr: jdoe123/eda-and-lgbm-baseline, msmith/tabnet-stacking-blueprint"`.

## 3. CV-first selection, never public-LB-only

The single most common shake-up trap: pick final submissions because they top the public LB. Public LB is computed on a small slice (often <30%) of the test set; private LB is the rest. Public-LB tops routinely fall hundreds of places on private LB.

**Operational rule:**

- `stage2_modeling/leaderboard.csv` tracks both `cv_score` and `public_lb` for every run.
- A candidate is "trustworthy" only if `|cv_score - public_lb_per_size_factor| < threshold`. The threshold comes from `comp_profile.yaml` (typically 1× CV std).
- `stage3_submit/recommendations.md` ranks candidates by **trust-adjusted CV**, never raw public LB.
- For the 2 final submissions (deadline-side, user-picked), the skill **must** present one "safe" choice (highest trustworthy CV) and one "ambitious" choice (best risk-adjusted ensemble). User picks; skill never silently picks both as gold-rush.

## 4. No LB probing

LB probing means submitting near-duplicate predictions to back-infer signal from public LB (e.g., toggling one row at a time, encoding test-set indices in submission noise). Kaggle treats this as cheating and has DQ'd participants over it.

**Operational rule:** Two consecutive submissions whose predictions differ by less than 1e-6 mean absolute distance are flagged by the orchestrator and **blocked** unless the user explicitly bypasses with `--allow-near-duplicate` and writes a one-line justification into `stage3_submit/recommendations.md`.

## 5. Single account, no team violations

Kaggle's one-account-per-person rule is enforced by IP / payment / browser fingerprinting. Using multiple accounts to fake-team or to extend submission quota is an immediate ban for all accounts.

**Operational rule:** `run.yaml` records exactly one `kaggle_username`. If the user provides a second username mid-run, the skill refuses and escalates.

If the competition allows teams and the user is on one, `run.yaml.team` must list all member usernames, and `submission_log.jsonl` records which teammate's account was used for each submission. Teammates must agree on attribution.

## 6. Quota honesty — no empty / random submissions to burn quota

Don't submit random or duplicate-of-yesterday predictions just to "use" a submission. This is a waste of the public LB feedback channel and the skill loses calibration.

**Operational rule:** Every submission requires a non-trivial CV score logged in `stage2_modeling/leaderboard.csv` (CV must be better than the median-prediction baseline for the metric, or have a documented diversification rationale in `recommendations.md`).

## 7. Honest CV — never tune the split to match LB

If your CV says 0.81 but public LB says 0.78, the temptation is to "fix" the CV by changing the split until it matches. This is overfitting to the LB through the back door and guarantees a private-LB collapse.

**Operational rule:** `stage0_bootstrap/cv_split.yaml` (decided at Stage 0 with rationale) may only be changed by the user, never by the agent. If the agent thinks the split is wrong, it escalates with a written justification; user has to approve.

## 8. Compute budget gate

Before any training run, the skill reads `compute_env.yaml` and `comp_profile.yaml`, estimates wallclock + GPU-hours, and refuses to start runs that exceed the user's stated budget. Silent budget overruns are how runs get killed mid-fold and produce useless artifacts.

## 9. Deadline awareness — no eleventh-hour panic submissions

In the last 24 hours before deadline, the skill switches to "deadline mode":

- No new model architectures, only finishing in-flight runs and ensembling existing OOFs.
- Reserves at least 1 submission for the final selection step.
- Forces the user-pick gate for the final 2 submissions (see Rule 3).

## 10. External data must be Kaggle-shared

Many competitions require any external dataset to be (a) publicly available, (b) shared on Kaggle as a dataset before the competition deadline. Using private external data is a DQ.

**Operational rule:** Anything beyond `data/raw/` (which is `kaggle competitions download` output) goes through `stage0_bootstrap/external_data.yaml`, listing source URL + Kaggle dataset ID + license + when it was shared. No external data without that entry.

---

## When in doubt, escalate

If a proposed action sits in a gray area for any of these rules, the skill **stops** and writes to `stage3_submit/recommendations.md` with the proposed action, the rule it might bump against, and a request for the user to approve or deny. The default is: **do not act**.
