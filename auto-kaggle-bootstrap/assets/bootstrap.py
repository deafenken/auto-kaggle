"""Stage 0 — bootstrap entry point.

Usage (called by the skill or directly by the supervisor):

    python bootstrap.py <comp_slug> --runs-dir runs/

Prerequisites:
    - kaggle CLI installed and authenticated
    - runs/<slug>/run.yaml already exists, written by the orchestrator
      with comp_slug, url, kaggle_username, target_tier, supervisor mode

The script does the side-effects defined in SKILL.md. It is intentionally
non-interactive — interactive Q&A happens inside the agent, before this
runs. By the time bootstrap.py is invoked, run.yaml has the 4 answers.

This module is idempotent: re-running on a finished bootstrap is a no-op
(see step 0). To force a re-bootstrap, delete stage0_bootstrap/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Local import — kaggle_helpers lives next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_helpers import (  # noqa: E402
    KaggleAuthError,
    download_competition_data,
    view_competition,
    write_heartbeat,
)


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    tmp.replace(path)


def _append_progress(progress_log: Path, event: dict[str, Any]) -> None:
    """Append-only progress event. Mirrors the helper used by every other stage
    so resume-by-default works without a shared utils module."""
    import json as _json  # local: keep top-level imports minimal
    from datetime import datetime as _dt, timezone as _tz

    progress_log.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts_utc": _dt.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **event}
    with progress_log.open("a") as f:
        f.write(_json.dumps(event) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def parse_competition_view(raw: str) -> dict[str, Any]:
    """The `kaggle competitions view` output is a key:value text block, not strict YAML.

    We tolerate it. Lines look like `Title:      <value>` or `Deadline:   2026-08-15 23:59 UTC`.
    """
    out: dict[str, Any] = {}
    for line in raw.splitlines():
        m = re.match(r"^([A-Za-z][A-Za-z ]+):\s+(.+)$", line.strip())
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            out[key] = m.group(2).strip()
    return out


def detect_task_type_from_submission(sample_submission_path: Path) -> str | None:
    """Cheap first-pass detection from the sample submission's columns.

    Returns one of the canonical task types, or None if undetermined.
    The full ruleset lives in references/task-type-detection.md; this function
    handles only the most common patterns and leaves harder cases to the agent.
    """
    if not sample_submission_path.exists():
        return None
    head = sample_submission_path.read_text(errors="ignore").splitlines()[:3]
    if not head:
        return None
    header = [c.strip().lower() for c in head[0].split(",")]
    if header == ["id", "target"]:
        if len(head) >= 2:
            try:
                val = float(head[1].split(",")[1])
                if val in (0.0, 1.0):
                    return "tabular-binary"
            except (ValueError, IndexError):
                pass
        return "tabular-regression"
    if "predicted" in header and len(header) == 2:
        return "image-segmentation"  # heuristic — RLE strings live here
    if {"series_id", "step", "event", "score"}.issubset(header):
        return "time-series-event-detection"
    if len(header) > 2 and header[0] in ("id", "image_id", "sample_id"):
        # one column per class
        return "tabular-multiclass"
    return None


def is_bootstrap_done(stage_dir: Path) -> bool:
    """A bootstrap is "done" only after the agent has finalized it.

    bootstrap.py writes a *partial* profile (it cannot WebFetch the rules page
    itself; the agent does Step 7 of the SKILL.md workflow). We refuse to skip
    until all of:
      - comp_profile.yaml exists AND `bootstrap_partial: false`
      - rules_summary.md exists
      - compute_env.yaml exists
    """
    cp_path = stage_dir / "comp_profile.yaml"
    if not cp_path.exists():
        return False
    if not (stage_dir / "rules_summary.md").exists():
        return False
    if not (stage_dir / "compute_env.yaml").exists():
        return False
    try:
        cp = yaml.safe_load(cp_path.read_text()) or {}
    except yaml.YAMLError:
        return False
    if cp.get("bootstrap_partial") is True:
        return False
    return True


def bootstrap(comp_slug: str, runs_dir: Path) -> int:
    run_dir = runs_dir / comp_slug
    if not run_dir.exists():
        sys.stderr.write(
            f"run_dir does not exist: {run_dir}\n"
            "The orchestrator must create runs/<slug>/run.yaml before bootstrap runs.\n"
        )
        return 2

    run_yaml_path = run_dir / "run.yaml"
    if not run_yaml_path.exists():
        sys.stderr.write(f"run.yaml missing at {run_yaml_path}\n")
        return 2

    run_cfg = yaml.safe_load(run_yaml_path.read_text())
    if run_cfg.get("comp_slug") != comp_slug:
        sys.stderr.write(
            f"run.yaml comp_slug mismatch: yaml says {run_cfg.get('comp_slug')}, "
            f"argv says {comp_slug}\n"
        )
        return 2

    stage_dir = run_dir / "stage0_bootstrap"
    progress_log = run_dir / "progress.jsonl"

    # Step 0 — idempotency
    if is_bootstrap_done(stage_dir):
        write_heartbeat(run_dir, "stage0", "already_done")
        print(f"bootstrap already done for {comp_slug}; nothing to do")
        return 0

    write_heartbeat(run_dir, "stage0", "view_competition")

    # Step 2 — confirm comp exists
    try:
        raw_view = view_competition(comp_slug, progress_log_path=progress_log)
    except KaggleAuthError as e:
        sys.stderr.write(f"ESCALATE: kaggle auth failed: {e}\n")
        return 3

    # `kaggle competitions view` output is a `Key: Value` text block, not JSON.
    # Name accordingly so a future reader doesn't expect JSON.
    raw_view_path = stage_dir / "raw_comp_view.txt"
    _atomic_write_text(raw_view_path, raw_view)

    parsed_view = parse_competition_view(raw_view)
    # Step 3 — refuse non-medal competitions
    category = parsed_view.get("category", "").lower()
    title = parsed_view.get("title", comp_slug)
    if "knowledge" in category or "tutorial" in category:
        msg = (
            f"# Bootstrap escalation — {title}\n\n"
            f"This competition's category is `{parsed_view.get('category', '?')}`, which "
            f"does not award medals. The auto-kaggle pipeline is designed for medal-hunting.\n\n"
            f"If you want to use this as practice, run the orchestrator with `--practice`.\n"
        )
        _atomic_write_text(stage_dir / "hand_off.md", msg)
        return 4

    # Step 5 — download data
    write_heartbeat(run_dir, "stage0", "downloading_data")
    data_raw = run_dir / "data" / "raw"
    try:
        total_bytes = download_competition_data(
            comp_slug, data_raw, progress_log_path=progress_log
        )
    except KaggleAuthError as e:
        msg = (
            f"# Bootstrap escalation — data download failed\n\n"
            f"`kaggle competitions download -c {comp_slug}` returned auth error: {e}\n\n"
            f"Most common cause: you have not accepted the competition rules on the website. "
            f"Visit https://www.kaggle.com/competitions/{comp_slug} and accept, then resume.\n"
        )
        _atomic_write_text(stage_dir / "hand_off.md", msg)
        return 4

    # Step 4 — basic task-type detection
    write_heartbeat(run_dir, "stage0", "detecting_task_type")
    sample_sub = data_raw / "sample_submission.csv"
    task_type = detect_task_type_from_submission(sample_sub)

    # Step 8 — comp_profile.yaml (partial — agent fills in the rest after rules parsing)
    comp_profile: dict[str, Any] = {
        "comp_slug": comp_slug,
        "title": title,
        "category": parsed_view.get("category", "unknown"),
        "deadline_utc": parsed_view.get("deadline"),
        "task_type": task_type,  # may be None if heuristic failed
        "metric": {
            "name": parsed_view.get("evaluationmetric", "unknown"),
            "direction": None,  # agent fills in
            "description": None,
            "evaluation_script_path": None,
        },
        "submission": {
            "format": "csv",
            "columns": _read_columns_if_exists(sample_sub),
            "size_limit_mb": None,
            "code_only": False,  # agent verifies via rules parsing
        },
        "quota": {
            "daily_limit": 5,  # agent overwrites from rules page
            "resets_at_utc": "00:00",
            "final_selections": 2,
        },
        "team": {
            "max_size": None,
            "solo_required": False,
        },
        "external_data": {
            "allowed": None,
            "must_be_shared": None,
            "pretrained_cutoff_utc": None,
        },
        "bootstrap_partial": True,
        "_note": (
            "This profile was written by bootstrap.py and is partial. "
            "The agent must finish it after parsing rules_summary.md "
            "(see auto-kaggle-bootstrap/references/rules-parsing.md)."
        ),
    }
    _atomic_write_yaml(stage_dir / "comp_profile.yaml", comp_profile)
    _append_progress(
        progress_log,
        {
            "stage": "stage0",
            "event": "comp_profile_written",
            "task_type": task_type,
            "deadline_utc": parsed_view.get("deadline"),
            "partial": True,
        },
    )

    # Quick data stats placeholder — agent fills the full file
    stats_lines = [f"# Data stats — {comp_slug}\n\n"]
    for p in sorted(data_raw.glob("*")):
        if p.is_file():
            stats_lines.append(f"- `{p.name}` — {p.stat().st_size:,} bytes\n")
    _atomic_write_text(stage_dir / "data_stats.md", "".join(stats_lines))

    # Hand-off briefing — minimal, agent finalizes
    handoff = (
        f"# Stage 0 → Stage 1 hand-off (partial — agent must complete)\n\n"
        f"## What I did\n"
        f"- Downloaded {total_bytes:,} bytes of data to `runs/{comp_slug}/data/raw/`.\n"
        f"- Detected task type (heuristic): `{task_type or 'UNKNOWN — agent must complete'}`.\n"
        f"- Wrote partial comp_profile.yaml. Fields with value `null` need agent attention.\n\n"
        f"## What's true now\n"
        f"- Deadline (UTC): `{parsed_view.get('deadline', 'UNKNOWN')}`\n"
        f"- Metric (raw): `{parsed_view.get('evaluationmetric', 'UNKNOWN')}` — agent must "
        f"determine direction and canonical name.\n"
        f"- Submission format: heuristic says CSV; agent must verify code-only constraints.\n"
        f"- External data rules: NOT YET PARSED. Agent must fetch rules page.\n\n"
        f"## What you should do next\n"
        f"Agent: fetch the comp's Overview / Rules / Evaluation pages, parse per "
        f"references/rules-parsing.md, complete `comp_profile.yaml` (especially "
        f"`metric.direction`, `quota.daily_limit`, `external_data.*`, "
        f"`team.*`, and `submission.code_only`), then flip "
        f"`bootstrap_partial: false`, and rewrite this hand_off in final form.\n"
    )
    _atomic_write_text(stage_dir / "hand_off.md", handoff)

    _append_progress(
        progress_log,
        {
            "stage": "stage0",
            "event": "bootstrap_script_done",
            "note": "partial profile written; agent must complete rules parsing + compute_env",
        },
    )

    write_heartbeat(run_dir, "stage0", "done_pending_agent_finalize")
    print("bootstrap.py done — partial profile written, agent must finalize")
    return 0


def _read_columns_if_exists(path: Path) -> list[str]:
    if not path.exists():
        return []
    first = path.read_text(errors="ignore").splitlines()
    if not first:
        return []
    return [c.strip() for c in first[0].split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("comp_slug")
    ap.add_argument("--runs-dir", default="runs", help="parent directory of comp run folders")
    args = ap.parse_args()
    return bootstrap(args.comp_slug, Path(args.runs_dir))


if __name__ == "__main__":
    sys.exit(main())
