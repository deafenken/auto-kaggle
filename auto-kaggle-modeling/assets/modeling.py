"""Stage 2 — modeling dispatcher.

Sets up a new run directory by copying a template, writes config.yaml +
attribution.md, and invokes the template's train.py as a subprocess. Does NOT
do any heavy ML in-process — that lives in the templates.

This script is intentionally thin. The "decide what to run next" logic in
SKILL.md (priority queue over ideas_pool.md) is the agent's job, not this
script's. By the time modeling.py is invoked, the agent has already decided:

    python modeling.py <comp_slug> \
        --run-id 2026-05-12-lgbm-baseline \
        --template tabular-lgbm \
        --idea-keys cv:stratified-kfold-5,feature:log1p_target_then_xgboost \
        --config-overrides seeds=[42],model_params.num_leaves=63

Usage:
    python modeling.py <comp_slug> \
        --run-id <id> \
        --template <name> \
        --idea-keys <comma-separated keys from ideas_pool.md> \
        [--attribution-cite <comma-separated kernel refs>] \
        [--budget-hours <float>] \
        [--dry-run]

Exit codes:
    0  run completed (or already complete on retry)
    2  bad args / missing files
    3  budget gate refused (escalation needed)
    4  training subprocess failed
    5  resume detected; partial completion ok, will continue next invocation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "auto-kaggle-bootstrap" / "assets"))

from kaggle_helpers import write_heartbeat  # noqa: E402
from leaderboard import (  # noqa: E402
    record_run_completed,
    record_run_failed,
    record_run_start,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_progress(progress_log: Path, event: dict[str, Any]) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts_utc": _now_iso(), **event}
    with progress_log.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _hash_dir(d: Path) -> str:
    """Stable hash of a directory's files (sorted), 8 chars."""
    h = hashlib.sha1()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(d).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:8]


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse repeated `--config-override key=value` flags into a nested dict.

    Supported syntax: dotted keys like `model_params.num_leaves=63`,
    JSON values for lists / dicts, plain scalars otherwise.
    """
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        try:
            val = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            val = raw
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = val
    return out


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _budget_estimate_hours(
    template_yaml: dict,
    comp_profile: dict,
    cv_split: dict,
    seed_count: int,
    compute_env: dict | None,
    idea_keys: list[str],
    train_csv_path: Path,
) -> float:
    """Wallclock estimate per `references/budget-estimator.md`.

    estimate = base_h × n_folds × seed_count × data_size_factor × cost_multiplier
             × hardware_factor × long_seq_penalty × 1.3 (safety pad)
    """
    task_type = comp_profile.get("task_type") or "tabular-regression"
    base_h = float(template_yaml.get("base_hours_per_fold", {}).get(task_type, 1.0))
    n_folds = int(cv_split.get("n_folds", 5))

    # data_size_factor: tabular → rows / 1M, vision → images / 100k, NLP → tokens / 100M.
    # Without a measured count, fall back to file size as a proxy.
    data_size_factor = 1.0
    try:
        if train_csv_path.exists():
            if task_type.startswith("tabular"):
                # Approximate rows by line count for speed (no need to parse).
                with train_csv_path.open("rb") as f:
                    n_lines = sum(1 for _ in f)
                data_size_factor = max(1.0, n_lines / 1_000_000)
            elif task_type.startswith("image"):
                # No reliable image count from a CSV alone; use file row count as proxy.
                with train_csv_path.open("rb") as f:
                    n_lines = sum(1 for _ in f)
                data_size_factor = max(1.0, n_lines / 100_000)
            elif task_type.startswith("nlp"):
                # Rough: bytes / 4 ≈ tokens, then / 1e8.
                data_size_factor = max(1.0, train_csv_path.stat().st_size / 4 / 100_000_000)
    except OSError:
        data_size_factor = 1.0
    data_size_factor = min(data_size_factor, 5.0)  # cap per spec

    # cost_multiplier: pick the heaviest cost among the chosen ideas if recorded.
    # ideas_pool.md is not parsed here — keep as a configurable default. The
    # agent can pass `cost_multiplier` via config_overrides.budget.cost_multiplier
    # when planning a known-heavy run.
    cost_multiplier = 1.0
    if any("ensemble" in k or "stacking" in k for k in idea_keys):
        cost_multiplier = 0.2  # ensembles are cheap; small fraction of base
    elif any(":pseudo" in k or "domain-adaptive" in k for k in idea_keys):
        cost_multiplier = 2.5  # heavy known patterns

    # hardware_factor: from compute_env.
    hardware_factor = 1.0
    if compute_env:
        env_name = compute_env.get("env")
        if env_name == "kaggle-notebook":
            hardware_factor = 0.5
        elif env_name == "cpu-only" and not task_type.startswith("tabular"):
            hardware_factor = 2.0

    # long-sequence penalty for NLP.
    long_seq_penalty = 1.0
    if task_type.startswith("nlp"):
        max_len = 512
        # caller can override via template defaults or config_overrides
        # (we don't have it explicitly here; conservatively pad if unknown)
        long_seq_penalty = max(1.0, (max_len / 512) ** 1.6)

    return (
        base_h
        * n_folds
        * seed_count
        * data_size_factor
        * cost_multiplier
        * hardware_factor
        * long_seq_penalty
        * 1.3
    )


def _check_external_data_approved(comp_dir: Path) -> tuple[bool, str]:
    """Rule 10: before training, verify every external dataset referenced by
    recon has `Approved for use: YES` in `stage1_recon/external_data_candidates.md`.

    Returns `(all_approved, message)`. If the file does not exist (no externals
    referenced yet), returns `(True, "")`.
    """
    path = comp_dir / "stage1_recon" / "external_data_candidates.md"
    if not path.exists():
        return True, ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    unapproved: list[str] = []
    current_section: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            current_section = s[3:].strip()
            continue
        if not current_section:
            continue
        # Tolerate `- **Approved for use:** ...` (list-item form, the documented
        # one) as well as a bare `**Approved for use:** ...` line.
        s_low = s.lower().lstrip("- ").strip()
        if s_low.startswith("**approved for use:**"):
            verdict = s.split(":", 1)[1].strip().split()[0].upper()
            if verdict != "YES":
                unapproved.append(current_section)
    if unapproved:
        return False, (
            "external_data_candidates.md has these datasets NOT approved for use: "
            + ", ".join(unapproved)
            + ". Stage 2 refuses to train until the user reviews them. Set "
            "`Approved for use: YES` next to each entry to allow."
        )
    return True, ""


def _is_run_already_complete(run_dir: Path) -> bool:
    return (run_dir / "cv_score.json").exists() and (run_dir / "test_preds.csv").exists()


def setup_and_train(
    comp_slug: str,
    runs_dir: Path,
    run_id: str,
    template_name: str,
    idea_keys: list[str],
    attribution_cite: list[str],
    config_overrides: dict[str, Any],
    budget_hours: float | None,
    dry_run: bool,
) -> int:
    comp_dir = runs_dir / comp_slug
    if not comp_dir.exists():
        sys.stderr.write(f"run_dir does not exist: {comp_dir}\n")
        return 2

    stage_dir = comp_dir / "stage2_modeling"
    progress_log = comp_dir / "progress.jsonl"
    leaderboard_path = stage_dir / "leaderboard.csv"
    run_dir = stage_dir / "runs" / run_id

    # Load upstream contracts.
    comp_profile_path = comp_dir / "stage0_bootstrap" / "comp_profile.yaml"
    if not comp_profile_path.exists():
        sys.stderr.write(f"comp_profile.yaml missing: {comp_profile_path}\n")
        return 2
    comp_profile = yaml.safe_load(comp_profile_path.read_text())

    cv_split_path = stage_dir / "cv_split.yaml"
    if not cv_split_path.exists():
        sys.stderr.write(
            f"cv_split.yaml missing: {cv_split_path}\n"
            "Stage 2 Step 0 (CV scheme decision) must run before any training "
            "(integrity rule 7).\n"
        )
        return 2
    cv_split = yaml.safe_load(cv_split_path.read_text())

    template_src = HERE / "templates" / template_name
    if not template_src.exists():
        sys.stderr.write(f"template not found: {template_src}\n")
        return 2
    template_yaml = yaml.safe_load(
        (template_src / "template.yaml").read_text()
    ) if (template_src / "template.yaml").exists() else {}

    # Load compute_env for budget + capability checks.
    compute_env: dict | None = None
    ce_path = comp_dir / "stage0_bootstrap" / "compute_env.yaml"
    if ce_path.exists():
        compute_env = yaml.safe_load(ce_path.read_text())

    # Rule 10 gate — refuse to train if recon listed external datasets the user
    # has not yet approved. Cheap check, runs before we copy files.
    ok, msg = _check_external_data_approved(comp_dir)
    if not ok:
        sys.stderr.write(f"ESCALATE rule 10: {msg}\n")
        _append_progress(
            progress_log,
            {"stage": "stage2", "event": "external_data_block", "run_id": run_id, "reason": msg[:200]},
        )
        return 3

    write_heartbeat(comp_dir, "stage2", f"setup_run {run_id}")

    # Idempotency — if the run is already complete, just touch the leaderboard.
    if run_dir.exists() and _is_run_already_complete(run_dir):
        print(f"run {run_id} already complete; skipping")
        _append_progress(
            progress_log, {"stage": "stage2", "event": "run_skipped", "run_id": run_id}
        )
        return 0

    # Copy template (or use existing run dir on resume).
    if not run_dir.exists():
        shutil.copytree(template_src, run_dir)
    else:
        # Resume case — re-copy any files missing from a half-set-up run.
        for src in template_src.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(template_src)
            dst = run_dir / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    template_hash = _hash_dir(template_src)

    # Copy the shared CV/metric helper next to train.py so the template's
    # `from _cv_common import ...` works at runtime. This sidesteps the prior
    # broken `from ../tabular-lgbm/train` import chain.
    cv_common_src = HERE / "templates" / "_cv_common.py"
    if cv_common_src.exists():
        shutil.copy2(cv_common_src, run_dir / "_cv_common.py")

    # Build config.yaml from template defaults + overrides.
    base_cfg = template_yaml.get("defaults", {})
    merged = _deep_merge(
        base_cfg,
        {
            "run_id": run_id,
            "comp_slug": comp_slug,
            "template": template_name,
            "template_source_path": str(template_src.relative_to(HERE.parent.parent)),
            "template_version_hash": template_hash,
            "task_type": comp_profile.get("task_type"),
            "metric": comp_profile.get("metric"),
            "cv_split": cv_split,
            "idea_keys": idea_keys,
            # Critical: write ABSOLUTE paths. The training subprocess runs with
            # cwd=run_dir, so any relative path would resolve under that dir
            # (which is wrong — data lives at runs/<comp_slug>/data/raw/).
            "data_paths": {
                "train": str((comp_dir / "data" / "raw" / "train.csv").resolve()),
                "test": str((comp_dir / "data" / "raw" / "test.csv").resolve()),
                "sample_submission": str(
                    (comp_dir / "data" / "raw" / "sample_submission.csv").resolve()
                ),
            },
            "submission_format": comp_profile.get("submission", {}),
        },
    )
    merged = _deep_merge(merged, config_overrides)
    if "seeds" not in merged:
        merged["seeds"] = [42]

    # Budget gate.
    seed_count = len(merged.get("seeds", [42]))
    estimate_h = _budget_estimate_hours(
        template_yaml,
        comp_profile,
        cv_split,
        seed_count,
        compute_env,
        idea_keys,
        (comp_dir / "data" / "raw" / "train.csv").resolve(),
    )
    if budget_hours is None:
        if compute_env:
            budget_hours = float(
                (compute_env.get("constraints") or {}).get("max_wallclock_per_run_hours", 24)
            )
        else:
            budget_hours = 24.0
    if estimate_h > budget_hours:
        msg = (
            f"ESCALATE rule 8: estimated {estimate_h:.2f}h exceeds budget "
            f"{budget_hours:.2f}h for run {run_id}. Refuse to start.\n"
            f"Options: reduce seeds, reduce folds, switch template, raise budget.\n"
        )
        sys.stderr.write(msg)
        _append_progress(
            progress_log,
            {
                "stage": "stage2",
                "event": "budget_gate_refused",
                "run_id": run_id,
                "estimate_h": estimate_h,
                "budget_h": budget_hours,
            },
        )
        # Persist a partial config so the user can see the proposed setup.
        _atomic_write(run_dir / "config.yaml", yaml.safe_dump(merged, sort_keys=False))
        return 3

    merged["budget_hours"] = budget_hours
    merged["budget_estimate_hours"] = estimate_h
    _atomic_write(run_dir / "config.yaml", yaml.safe_dump(merged, sort_keys=False))

    # Attribution.
    attribution_lines = [
        f"# Attribution — {run_id}\n",
        "",
        "## Ideas used (from ideas_pool.md)",
    ]
    for k in idea_keys:
        attribution_lines.append(f"- `{k}`")
    if attribution_cite:
        attribution_lines += ["", "## Citations (Kaggle kernels)"]
        for c in attribution_cite:
            attribution_lines.append(f"- {c}")
    attribution_lines += [
        "",
        "## Own additions (not from any kernel)",
        "- _agent fills in after training, or 'None' if purely an ablation_",
        "",
        "## After-run notes (filled by agent after CV is in)",
        "- _filled after training_",
        "",
    ]
    _atomic_write(run_dir / "attribution.md", "\n".join(attribution_lines))

    if not attribution_cite and "+own" not in " ".join(idea_keys):
        # Rule 2 — at least one of cite or +own must be present.
        sys.stderr.write(
            "ESCALATE rule 2: no kernel citations AND no +own marker — "
            "every run must have at least one of these to remain submittable.\n"
        )
        return 3

    record_run_start(leaderboard_path, run_id, template_name, idea_keys)
    _append_progress(
        progress_log,
        {
            "stage": "stage2",
            "event": "run_started",
            "run_id": run_id,
            "template": template_name,
            "estimate_h": estimate_h,
        },
    )

    if dry_run:
        print(f"dry-run OK: run {run_id} set up, would train for ~{estimate_h:.2f}h")
        return 0

    # Invoke training subprocess. cwd=run_dir so the template can `from
    # _cv_common import ...` (sibling file we just copied). Pass absolute
    # paths for --run-dir and --progress-log so the template never has to
    # resolve them against an unstable cwd.
    train_py = run_dir / "train.py"
    if not train_py.exists():
        sys.stderr.write(f"template missing train.py: {train_py}\n")
        return 2
    run_dir_abs = run_dir.resolve()
    progress_log_abs = progress_log.resolve()
    cmd = [
        sys.executable,
        "train.py",                       # basename, since cwd=run_dir
        "--config",
        "config.yaml",
        "--run-dir",
        str(run_dir_abs),
        "--progress-log",
        str(progress_log_abs),
    ]
    log_path = run_dir / "train.log"
    write_heartbeat(comp_dir, "stage2", f"training {run_id}")
    with log_path.open("ab") as logf:
        logf.write(f"\n--- training started {_now_iso()} ---\n".encode())
        proc = subprocess.run(cmd, cwd=run_dir_abs, stdout=logf, stderr=subprocess.STDOUT)
        logf.write(f"\n--- training exit {proc.returncode} {_now_iso()} ---\n".encode())

    if proc.returncode != 0:
        record_run_failed(leaderboard_path, run_id, reason="failed")
        _append_progress(
            progress_log,
            {"stage": "stage2", "event": "run_failed", "run_id": run_id, "exit": proc.returncode},
        )
        sys.stderr.write(
            f"training failed for {run_id} (exit {proc.returncode}); see {log_path}\n"
        )
        return 4

    # Read cv_score.json and update leaderboard.
    cv_path = run_dir / "cv_score.json"
    if not cv_path.exists():
        sys.stderr.write(
            f"training claimed success but {cv_path} is missing — partial run?\n"
        )
        return 5
    cv = json.loads(cv_path.read_text())
    record_run_completed(
        leaderboard_path,
        run_id,
        cv_metric=cv["metric"],
        cv_score=float(cv["mean"]),
        cv_std=float(cv.get("std", 0)),
        attribution_keys=attribution_cite or [],
        template=template_name,
        idea_keys=idea_keys,
    )
    _append_progress(
        progress_log,
        {
            "stage": "stage2",
            "event": "run_finished",
            "run_id": run_id,
            "cv_score": cv["mean"],
            "cv_std": cv.get("std", 0),
        },
    )

    _write_stage2_handoff(
        stage_dir=stage_dir,
        run_id=run_id,
        cv=cv,
        leaderboard_path=leaderboard_path,
        comp_profile=comp_profile,
        compute_env=compute_env,
        idea_keys=idea_keys,
        attribution_cite=attribution_cite,
    )

    write_heartbeat(comp_dir, "stage2", f"done {run_id}")
    print(
        f"run {run_id} done: {cv['metric']} = {cv['mean']:.6f} "
        f"(std {cv.get('std', 0):.6f})"
    )
    return 0


def _write_stage2_handoff(
    *,
    stage_dir: Path,
    run_id: str,
    cv: dict,
    leaderboard_path: Path,
    comp_profile: dict,
    compute_env: dict | None,
    idea_keys: list[str],
    attribution_cite: list[str],
) -> None:
    """Stage 2 → Stage 3 hand-off (`SKILL.md` step 7)."""
    from leaderboard import best_by_cv, load  # noqa: PLC0415

    metric_name = cv.get("metric", "metric")
    direction = (comp_profile.get("metric") or {}).get("direction", "maximize")
    rows = load(leaderboard_path)
    n_runs = len([r for r in rows if r.get("status") == "completed"])
    best = best_by_cv(leaderboard_path, direction=("max" if direction == "maximize" else "min"))
    best_score = best.get("cv_score") if best else "n/a"
    best_run = best.get("run_id") if best else "n/a"

    body = (
        f"# Stage 2 → Stage 3 hand-off (run {run_id} finished)\n\n"
        f"## What I did\n"
        f"- Trained `{run_id}` with idea_keys={idea_keys}.\n"
        f"- CV {metric_name} = {cv['mean']:.6f} (std {cv.get('std', 0):.6f}).\n"
        f"- Attribution kernels: {', '.join(attribution_cite) if attribution_cite else 'none (+own)'}\n\n"
        f"## What's true now\n"
        f"- Total completed runs: {n_runs}.\n"
        f"- Best CV by trust filter: {best_score} (run `{best_run}`).\n"
        f"- Compute env: `{(compute_env or {}).get('env', 'unknown')}`.\n\n"
        f"## What you should do next\n"
        f"Stage 3: refresh `recommendations.md` ranking by trust-adjusted CV. If "
        f"`{run_id}` beats the current best by more than 1× cv_std, it should appear "
        f"near the top. Verify quota with `kaggle competitions submissions` before submitting.\n"
    )
    _atomic_write(stage_dir / "hand_off.md", body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("comp_slug")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument(
        "--idea-keys",
        default="",
        help="comma-separated ideas_pool.md keys",
    )
    ap.add_argument(
        "--attribution-cite",
        default="",
        help="comma-separated kernel refs (author/kernel-slug)",
    )
    ap.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="dotted key=value or key=<json>; can repeat",
    )
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    return setup_and_train(
        comp_slug=args.comp_slug,
        runs_dir=Path(args.runs_dir),
        run_id=args.run_id,
        template_name=args.template,
        idea_keys=[k.strip() for k in args.idea_keys.split(",") if k.strip()],
        attribution_cite=[c.strip() for c in args.attribution_cite.split(",") if c.strip()],
        config_overrides=_parse_overrides(args.config_override),
        budget_hours=args.budget_hours,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
