"""Stage 3 — submit entry point.

Reconciles quota, refreshes recommendations, optionally submits a user-chosen
run, polls public LB, logs everything. See SKILL.md for the full workflow and
references/{ranking-rubric,quota-reconciliation,final-selection}.md for
details.

Usage:
    # Just refresh quota + recommendations, no submit:
    python submit.py <slug>

    # Quota only (skip recommendations refresh):
    python submit.py <slug> --quota-only

    # Submit a specific run:
    python submit.py <slug> --submit <run_id>

    # Override message:
    python submit.py <slug> --submit <run_id> --message "<text including attr:>"

    # Override near-duplicate block (Rule 4):
    python submit.py <slug> --submit <run_id> --allow-near-duplicate --reason "<why>"

    # Final-mode submission (last 6h before deadline):
    python submit.py <slug> --submit <run_id> --final-submission-confirm

Exit codes:
    0  done
    2  bad args / missing files
    3  pre-submit gate refused / escalation needed
    4  kaggle auth error
    5  partial — submission accepted but public LB not yet known
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "auto-kaggle-bootstrap" / "assets"))
sys.path.insert(0, str(HERE.parent.parent / "auto-kaggle-modeling" / "assets"))

from kaggle_helpers import (  # noqa: E402
    KaggleAuthError,
    list_submissions,
    submit_csv,
    write_heartbeat,
)
from leaderboard import record_submission_result  # noqa: E402
from near_duplicate import check_against_log  # noqa: E402
from quota import (  # noqa: E402
    clear_wait_until,
    fmt_until,
    read_wait_until,
    reconcile,
    write_wait_until,
)
from recommend import (  # noqa: E402
    _read_leaderboard,
    _read_submission_log,
    build_recommendations,
    write_recommendations,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_progress(progress_log: Path, event: dict[str, Any]) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts_utc": _now_iso(), **event}
    with progress_log.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _append_submission_log(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _read_attribution(attr_path: Path) -> tuple[list[str], bool]:
    """Return (citation refs, has_+own_marker)."""
    if not attr_path.exists():
        return [], False
    text = attr_path.read_text()
    citations: list[str] = []
    in_cit = False
    has_own = "+own" in text or "+ own" in text or "Own additions" in text and "None" not in text.split("Own additions")[1].split("##")[0]
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("## citations"):
            in_cit = True
            continue
        if in_cit and s.startswith("##"):
            break
        if in_cit and s.startswith("- "):
            ref = s[2:].split()[0].strip(",")
            if "/" in ref:
                citations.append(ref)
    return citations, has_own


def _read_cv(run_dir: Path) -> tuple[float, float, str]:
    """Return (cv_score, cv_std, metric_name) from cv_score.json."""
    cv = json.loads((run_dir / "cv_score.json").read_text())
    return float(cv["mean"]), float(cv.get("std", 0)), str(cv.get("metric") or "metric")


def _build_message(run_id: str, citations: list[str], has_own: bool, cv: float, std: float, metric: str) -> str:
    short_cv = f"CV {metric}={cv:.5f} (std {std:.5f})"
    cit_part = "attr: " + ", ".join(citations[:3]) if citations else ""
    own_part = "+ own" if has_own else ""
    msg = f"{run_id} — {short_cv}"
    if cit_part:
        msg += f" | {cit_part}"
    if own_part:
        msg += f" {own_part}"
    return msg


def _poll_public_lb(comp_slug: str, run_id: str, file_path: Path, max_wait_s: int = 600) -> float | None:
    """Poll `kaggle competitions submissions` until our submission shows a public score.

    Heuristic match: latest submission whose filename matches our basename.
    """
    target_name = file_path.name
    waited = 0
    backoffs = [30, 60, 120, 300]
    bi = 0
    while waited < max_wait_s:
        raw = list_submissions(comp_slug)
        # The verbose output includes filename and publicScore.
        for line in raw.splitlines():
            if target_name in line and "complete" in line.lower():
                # Try to extract publicScore (last numeric column).
                parts = line.split()
                for p in reversed(parts):
                    try:
                        val = float(p)
                        return val
                    except ValueError:
                        continue
        sleep = backoffs[min(bi, len(backoffs) - 1)]
        time.sleep(sleep)
        waited += sleep
        bi += 1
    return None


def _deadline_check(deadline_utc: str | None) -> tuple[bool, bool, int]:
    """Return (in_deadline_24h, in_deadline_6h, hours_to_deadline)."""
    if not deadline_utc:
        return False, False, 9999
    try:
        dl = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    except ValueError:
        return False, False, 9999
    now = datetime.now(timezone.utc)
    delta_h = (dl - now).total_seconds() / 3600
    return (delta_h <= 24), (delta_h <= 6), int(delta_h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("comp_slug")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--submit", help="run_id to submit")
    ap.add_argument("--quota-only", action="store_true")
    ap.add_argument("--message", help="override submission message")
    ap.add_argument("--allow-near-duplicate", action="store_true")
    ap.add_argument("--reason", default="")
    ap.add_argument("--final-submission-confirm", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    comp_slug = args.comp_slug
    run_dir = Path(args.runs_dir) / comp_slug
    stage_dir = run_dir / "stage3_submit"
    stage_dir.mkdir(parents=True, exist_ok=True)
    progress_log = run_dir / "progress.jsonl"

    # Load comp profile for metric direction, daily quota, deadline.
    cp_path = run_dir / "stage0_bootstrap" / "comp_profile.yaml"
    if not cp_path.exists():
        sys.stderr.write(f"comp_profile.yaml missing: {cp_path}\n")
        return 2
    cp = yaml.safe_load(cp_path.read_text())
    daily_limit = int((cp.get("quota") or {}).get("daily_limit") or 5)
    metric = cp.get("metric") or {}
    metric_dir = metric.get("direction") or "maximize"
    metric_name = metric.get("name") or "metric"
    deadline_utc = cp.get("deadline_utc")

    write_heartbeat(run_dir, "stage3", "reconcile_quota")

    # Wait-until check before doing anything else.
    wu = read_wait_until(stage_dir)
    now = datetime.now(timezone.utc)
    if wu is not None and now < wu:
        print(
            f"[auto-kaggle-submit] {comp_slug}  STILL_WAITING until "
            f"{wu.strftime('%Y-%m-%dT%H:%M:%SZ')} (in {fmt_until(wu)})"
        )
        return 0
    if wu is not None and now >= wu:
        clear_wait_until(stage_dir)
        _append_progress(progress_log, {"stage": "stage3", "event": "quota_reset"})

    # Reconcile quota with Kaggle.
    try:
        state, quota_line = reconcile(comp_slug, stage_dir, daily_limit, progress_log=progress_log)
    except KaggleAuthError as e:
        sys.stderr.write(f"ESCALATE auth: {e}\n")
        return 4
    print(f"[auto-kaggle-submit] {comp_slug}  {quota_line}")

    if state.exhausted:
        write_wait_until(stage_dir, state.next_reset_utc)
        _append_progress(progress_log, {"stage": "stage3", "event": "quota_exhausted"})
        print(f"WAIT_UNTIL {state.next_reset_utc} (in {fmt_until(wu or datetime.fromisoformat(state.next_reset_utc.replace('Z', '+00:00')))})")
        if args.submit:
            sys.stderr.write("ESCALATE: quota exhausted, cannot submit until reset\n")
        return 0 if not args.submit else 3

    if args.quota_only:
        write_heartbeat(run_dir, "stage3", "quota_only_done")
        return 0

    # Refresh recommendations.
    leaderboard = _read_leaderboard(run_dir / "stage2_modeling" / "leaderboard.csv")
    submissions = _read_submission_log(stage_dir / "submission_log.jsonl")
    text = build_recommendations(
        run_dir,
        leaderboard,
        submissions,
        quota_line=quota_line,
        metric_direction=metric_dir,
        alpha=args.alpha,
        metric_name=metric_name,
        deadline_utc=deadline_utc,
    )
    write_recommendations(run_dir, text)
    _append_progress(progress_log, {"stage": "stage3", "event": "recommendation_refreshed"})

    if not args.submit:
        _append_progress(progress_log, {"stage": "stage3", "event": "awaiting_user_pick"})
        print(
            "Refreshed runs/<slug>/stage3_submit/recommendations.md — "
            "tell me which candidate to submit (or run with --submit <run_id>)."
        )
        return 0

    # ----- Submission path -----
    run_id = args.submit
    run_run_dir = run_dir / "stage2_modeling" / "runs" / run_id
    preds_path = run_run_dir / "test_preds.csv"
    attr_path = run_run_dir / "attribution.md"
    if not preds_path.exists():
        sys.stderr.write(f"missing predictions: {preds_path}\n")
        return 2
    if not attr_path.exists():
        sys.stderr.write(f"missing attribution: {attr_path}\n")
        return 2

    citations, has_own = _read_attribution(attr_path)
    if not citations and not has_own:
        sys.stderr.write(
            "ESCALATE rule 2: attribution.md has no Citations and no +own marker. "
            "Refusing to submit.\n"
        )
        return 3

    cv, std, _metric = _read_cv(run_run_dir)

    # Deadline-mode gates
    in_24h, in_6h, hours_left = _deadline_check(deadline_utc)
    if in_6h and not args.final_submission_confirm:
        sys.stderr.write(
            f"ESCALATE rule 9: only {hours_left}h to deadline; pass "
            "--final-submission-confirm to proceed.\n"
        )
        return 3

    # Near-duplicate check
    log_path = stage_dir / "submission_log.jsonl"
    is_dup, min_mae, match_run = check_against_log(preds_path, log_path)
    if is_dup and not args.allow_near_duplicate:
        sys.stderr.write(
            f"ESCALATE rule 4: near-duplicate of {match_run} (MAE={min_mae:.2e}). "
            "Pass --allow-near-duplicate --reason \"<why>\" to override.\n"
        )
        return 3

    message = args.message or _build_message(run_id, citations, has_own, cv, std, metric_name)
    if args.allow_near_duplicate and args.reason:
        message += f" [override near-dup: {args.reason[:60]}]"

    # Final guard: Rule 2 — message must carry attribution.
    if "attr:" not in message and not has_own:
        sys.stderr.write("ESCALATE rule 2: built message has no attr: token\n")
        return 3

    write_heartbeat(run_dir, "stage3", f"submitting {run_id}")
    try:
        stdout = submit_csv(comp_slug, preds_path, message, progress_log_path=progress_log)
    except KaggleAuthError as e:
        sys.stderr.write(f"ESCALATE auth: {e}\n")
        return 4

    submission_id = None
    for line in stdout.splitlines():
        if "submission" in line.lower() and any(c.isdigit() for c in line):
            for tok in line.split():
                if tok.isdigit() and len(tok) >= 5:
                    submission_id = tok
                    break

    entry = {
        "ts_utc": _now_iso(),
        "run_id": run_id,
        "file": str(preds_path),
        "message": message,
        "kaggle_submission_id": submission_id,
        "public_lb": None,
        "cv_score": cv,
        "cv_std": std,
        "candidate_rank_at_time": None,
        "user_approved_by": args.submit,
        "is_final_candidate": bool(args.final_submission_confirm),
        "near_duplicate_override": bool(args.allow_near_duplicate),
        "near_duplicate_reason": args.reason if args.allow_near_duplicate else None,
    }
    _append_submission_log(log_path, entry)
    _append_progress(progress_log, {"stage": "stage3", "event": "submitted", "run_id": run_id})

    # Update quota cache locally (Kaggle reconcile next cycle will confirm).
    state.used_today += 1
    state.remaining = max(0, state.daily_limit - state.used_today)
    state.exhausted = state.used_today >= state.daily_limit
    write_heartbeat(run_dir, "stage3", "post_submit_poll_lb")

    # Poll for public LB (best-effort).
    print(f"submitted {run_id}; polling for public LB...")
    public_lb = _poll_public_lb(comp_slug, run_id, preds_path)
    if public_lb is not None:
        entry["public_lb"] = public_lb
        # Rewrite the last line — exception to append-only because the entry is incomplete.
        _rewrite_last_log_entry(log_path, entry)
        record_submission_result(
            run_dir / "stage2_modeling" / "leaderboard.csv",
            run_id,
            public_lb=public_lb,
            cv_score=cv,
        )
        _append_progress(
            progress_log,
            {"stage": "stage3", "event": "public_lb_known",
             "run_id": run_id, "public_lb": public_lb, "gap": abs(public_lb - cv)},
        )
        print(f"public LB: {public_lb:.6f} (gap {abs(public_lb - cv):.6f})")
        exit_code = 0
    else:
        print("public LB not available within poll window; will reconcile next cycle.")
        exit_code = 5

    # If now exhausted, write wait_until.
    if state.exhausted:
        write_wait_until(stage_dir, state.next_reset_utc)
        _append_progress(progress_log, {"stage": "stage3", "event": "quota_exhausted"})
        print(
            f"WAIT_UNTIL {state.next_reset_utc} "
            f"(quota exhausted; supervisor sleeps until then)"
        )

    write_heartbeat(run_dir, "stage3", "submit_done")
    return exit_code


def _rewrite_last_log_entry(path: Path, entry: dict) -> None:
    """Replace the last line of submission_log.jsonl with `entry`.

    Used once per submission, when public_lb finally lands. Append-only is
    relaxed for this single case because the original entry was deliberately
    incomplete (public_lb: null).
    """
    if not path.exists():
        _append_submission_log(path, entry)
        return
    lines = path.read_text().splitlines()
    if not lines:
        _append_submission_log(path, entry)
        return
    lines[-1] = json.dumps(entry)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    sys.exit(main())
