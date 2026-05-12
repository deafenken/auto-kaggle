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
    template_yaml: dict, comp_profile: dict, cv_split: dict, seed_count: int
) -> float:
    """Best-effort wallclock estimate. See references/budget-estimator.md."""
    task_type = comp_profile.get("task_type") or "tabular-regression"
    base_h = float(
        template_yaml.get("base_hours_per_fold", {}).get(task_type, 1.0)
    )
    n_folds = int(cv_split.get("n_folds", 5))
    # data_size_factor is rough — caller can refine via attribution.md later.
    data_size_factor = 1.0  # default; agent overrides if data_stats.md is informative
    cost_multiplier = 1.0
    hardware_factor = 1.0
    return base_h * n_folds * seed_count * data_size_factor * cost_multiplier * hardware_factor * 1.3


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
            "data_paths": {
                "train": str(comp_dir / "data" / "raw" / "train.csv"),
                "test": str(comp_dir / "data" / "raw" / "test.csv"),
                "sample_submission": str(comp_dir / "data" / "raw" / "sample_submission.csv"),
            },
            "submission_format": comp_profile.get("submission", {}),
        },
    )
    merged = _deep_merge(merged, config_overrides)
    if "seeds" not in merged:
        merged["seeds"] = [42]

    # Budget gate.
    seed_count = len(merged.get("seeds", [42]))
    estimate_h = _budget_estimate_hours(template_yaml, comp_profile, cv_split, seed_count)
    if budget_hours is None:
        # Try compute_env.yaml.
        ce_path = comp_dir / "stage0_bootstrap" / "compute_env.yaml"
        if ce_path.exists():
            ce = yaml.safe_load(ce_path.read_text()) or {}
            budget_hours = float(
                ce.get("constraints", {}).get("max_wallclock_per_run_hours", 24)
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

    # Invoke training subprocess.
    train_py = run_dir / "train.py"
    if not train_py.exists():
        sys.stderr.write(f"template missing train.py: {train_py}\n")
        return 2
    cmd = [
        sys.executable,
        str(train_py),
        "--config",
        "config.yaml",
        "--run-dir",
        str(run_dir),
        "--progress-log",
        str(progress_log),
    ]
    log_path = run_dir / "train.log"
    write_heartbeat(comp_dir, "stage2", f"training {run_id}")
    with log_path.open("ab") as logf:
        logf.write(f"\n--- training started {_now_iso()} ---\n".encode())
        proc = subprocess.run(cmd, cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT)
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
    write_heartbeat(comp_dir, "stage2", f"done {run_id}")
    print(
        f"run {run_id} done: {cv['metric']} = {cv['mean']:.6f} "
        f"(std {cv.get('std', 0):.6f})"
    )
    return 0


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
