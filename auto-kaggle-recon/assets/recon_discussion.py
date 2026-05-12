"""Stage 1 optional sub-step — pull and index the comp's Discussion threads.

Listing + metadata pulls go through `WebFetch` (the agent invokes this script as
a CLI driver, but the actual HTTP fetches are delegated — see SKILL.md). This
script handles:
  - the throttle window (separate from recon's own throttle, via last_discussion_at)
  - parsing the WebFetch-returned JSON of thread listings into discussion_index.json
  - per-thread full-body fetches for high-vote threads
  - idempotency: skip already-pulled threads unless --re-pull
  - clean append to ideas_pool.md / citations.bib happens in the agent (this script
    doesn't do distillation, same as recon.py)

Usage:
    python recon_discussion.py <comp_slug> [--runs-dir runs] [--force] [--re-pull]

Two modes:
    Mode A — full driver (default): the script orchestrates the listing parse
    + per-thread fetches, expecting the agent to have already saved
    `discussion_listings.json` and `discussions/<thread_id>/post.md` via
    WebFetch into the .discussion_tmp/ directory. This script is the
    aggregation + dedup + index update layer.

    Mode B — agent-mode: when invoked from inside an agent loop, the agent
    calls WebFetch directly and uses helper functions exported here
    (`merge_into_index`, `update_last_discussion_at`).

Exit codes:
    0  done (or throttled)
    2  bad args / missing run.yaml
    3  unrecoverable error (clock skew)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "auto-kaggle-bootstrap" / "assets"))
from kaggle_helpers import write_heartbeat  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_write_json(path: Path, data: Any) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


def _append_progress(progress_log: Path, event: dict[str, Any]) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts_utc": _now_iso(), **event}
    with progress_log.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _throttled(last_path: Path, interval_h: float, force: bool) -> tuple[bool, float]:
    if force or not last_path.exists():
        return False, 0.0
    try:
        text = last_path.read_text().strip().replace("Z", "+00:00")
        last_ts = datetime.fromisoformat(text)
    except (ValueError, OSError):
        return False, 0.0
    age_h = (_now_utc() - last_ts).total_seconds() / 3600
    if age_h < 0:
        return False, age_h
    return age_h < interval_h, age_h


def load_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return {e["thread_id"]: e for e in data}
    except (json.JSONDecodeError, KeyError):
        return {}


def write_index(path: Path, index: dict[str, dict[str, Any]]) -> None:
    entries = sorted(
        index.values(),
        key=lambda e: (-int(e.get("votes", 0)), e.get("posted_utc", "")),
    )
    _atomic_write_json(path, entries)


def merge_into_index(
    index: dict[str, dict[str, Any]],
    new_threads: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Return (updated_index, new_thread_ids, updated_thread_ids)."""
    new_ids: list[str] = []
    updated_ids: list[str] = []
    for t in new_threads:
        tid = str(t["thread_id"])
        if tid not in index:
            t["status"] = "new"
            t["last_pulled_utc"] = None
            t["distilled_at_utc"] = None
            t.setdefault("category", "other")
            t.setdefault("high_signal", False)
            index[tid] = t
            new_ids.append(tid)
            continue
        prev = index[tid]
        # If votes / replies moved, mark updated.
        if int(t.get("votes", 0)) > int(prev.get("votes", 0)) or int(
            t.get("replies", 0)
        ) > int(prev.get("replies", 0)):
            prev["votes"] = int(t.get("votes", prev.get("votes", 0)))
            prev["replies"] = int(t.get("replies", prev.get("replies", 0)))
            prev["status"] = "updated"
            updated_ids.append(tid)
        # Refresh title (in case author edited).
        prev["title"] = t.get("title") or prev.get("title")
    return index, new_ids, updated_ids


def threads_to_fetch_in_full(
    index: dict[str, dict[str, Any]], min_votes: int, cap: int, re_pull: bool
) -> list[dict[str, Any]]:
    """Pick threads worth fetching the full body for: above the vote threshold
    AND either new, updated, or never-distilled (or all if re-pull)."""
    out = []
    for tid, entry in index.items():
        if int(entry.get("votes", 0)) < min_votes:
            continue
        if re_pull:
            out.append(entry)
            continue
        if (
            entry.get("status") in ("new", "updated")
            or entry.get("distilled_at_utc") is None
        ):
            out.append(entry)
    out.sort(key=lambda e: -int(e.get("votes", 0)))
    return out[:cap]


def update_last_discussion_at(path: Path) -> None:
    _atomic_write_text(path, _now_iso())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("comp_slug")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--re-pull", action="store_true",
                    help="re-fetch every thread (default: only new/updated/un-distilled)")
    ap.add_argument("--min-votes", type=int, default=10,
                    help="ignore threads below this vote count")
    ap.add_argument("--cap", type=int, default=30,
                    help="max full-body fetches per cycle")
    args = ap.parse_args()

    run_dir = Path(args.runs_dir) / args.comp_slug
    if not run_dir.exists():
        sys.stderr.write(f"run_dir missing: {run_dir}\n")
        return 2

    run_yaml = run_dir / "run.yaml"
    if not run_yaml.exists():
        sys.stderr.write(f"run.yaml missing: {run_yaml}\n")
        return 2
    cfg = yaml.safe_load(run_yaml.read_text())
    rd_cfg = cfg.get("recon_discussion") or {}
    if not rd_cfg.get("enabled", False):
        print("discussion mining disabled in run.yaml.recon_discussion.enabled")
        return 0
    interval_h = float(rd_cfg.get("interval_hours", cfg.get("recon_interval_hours", 6)))

    stage_dir = run_dir / "stage1_recon"
    progress_log = run_dir / "progress.jsonl"
    last_path = stage_dir / "last_discussion_at"
    index_path = stage_dir / "discussion_index.json"
    tmp_dir = stage_dir / ".discussion_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    write_heartbeat(run_dir, "stage1_discussion", "throttle_check")
    throttled, age_h = _throttled(last_path, interval_h, args.force)
    if age_h < 0:
        sys.stderr.write(
            f"ESCALATE: system clock before last_discussion_at by {-age_h:.2f}h\n"
        )
        return 3
    if throttled:
        print(
            f"discussion throttled — last pull was {age_h:.1f}h ago, "
            f"next allowed in {interval_h - age_h:.1f}h"
        )
        return 0

    # The agent is responsible for calling WebFetch and dropping the
    # listing JSON at .discussion_tmp/listings.json before invoking this
    # script in driver mode. If that file is absent, the script writes a
    # one-line instruction to handoff_prompts/ and exits.
    listings_path = tmp_dir / "listings.json"
    if not listings_path.exists():
        handoff_dir = run_dir / "handoff_prompts"
        handoff_dir.mkdir(parents=True, exist_ok=True)
        prompt = (
            f"# Discussion listing fetch — {args.comp_slug}\n\n"
            f"Use WebFetch with URL "
            f"https://www.kaggle.com/competitions/{args.comp_slug}/discussion?sort=hotness "
            f"and the prompt template in "
            f"auto-kaggle-recon/references/discussion-mining.md to produce a JSON "
            f"array of thread metadata. Save the array to "
            f"`{listings_path.relative_to(run_dir)}` then re-invoke this script.\n"
        )
        _atomic_write_text(handoff_dir / "discussion_listings.md", prompt)
        print(
            f"HANDOFF_READY discussion_listings → {handoff_dir / 'discussion_listings.md'}\n"
            f"Run WebFetch per the handoff, save output to {listings_path}, then re-invoke."
        )
        return 0

    try:
        new_threads = json.loads(listings_path.read_text())
    except json.JSONDecodeError as e:
        sys.stderr.write(f"listings.json is malformed: {e}\n")
        return 2

    if not isinstance(new_threads, list):
        sys.stderr.write("listings.json must be a JSON array\n")
        return 2

    index = load_index(index_path)
    index, new_ids, updated_ids = merge_into_index(index, new_threads)

    write_heartbeat(run_dir, "stage1_discussion", "pick_full_fetches")
    to_fetch = threads_to_fetch_in_full(
        index, min_votes=args.min_votes, cap=args.cap, re_pull=args.re_pull
    )

    # Emit handoff prompts for the full-body fetches the agent must run.
    # We do this so the agent can batch them, rather than blocking here.
    if to_fetch:
        handoff_dir = run_dir / "handoff_prompts"
        handoff_dir.mkdir(parents=True, exist_ok=True)
        per_thread_dir = stage_dir / "discussions"
        per_thread_dir.mkdir(parents=True, exist_ok=True)
        prompt_lines = [
            f"# Discussion full-body fetches — {args.comp_slug}",
            "",
            f"For each of these {len(to_fetch)} threads, fetch the full body via WebFetch and",
            f"save the markdown to `stage1_recon/discussions/<thread_id>/post.md`:",
            "",
        ]
        for entry in to_fetch:
            tid = entry["thread_id"]
            prompt_lines.append(
                f"- `{tid}` — {entry.get('title', '(no title)')} — votes "
                f"{entry.get('votes', 0)}, posted {entry.get('posted_utc', '?')} — "
                f"URL: https://www.kaggle.com/competitions/{args.comp_slug}/discussion/{tid}"
            )
        prompt_lines += [
            "",
            "After saving, mark each thread's `last_pulled_utc` in",
            "`stage1_recon/discussion_index.json` (this script will not see them",
            "as pulled until you do).",
            "",
        ]
        _atomic_write_text(handoff_dir / "discussion_full_fetches.md", "\n".join(prompt_lines))

    write_index(index_path, index)
    update_last_discussion_at(last_path)
    _append_progress(
        progress_log,
        {
            "stage": "stage1",
            "event": "discussion_pulled",
            "threads_total": len(index),
            "new": len(new_ids),
            "updated": len(updated_ids),
            "queued_full_fetch": len(to_fetch),
        },
    )
    print(
        f"discussion_index updated: {len(new_ids)} new, {len(updated_ids)} updated, "
        f"{len(to_fetch)} queued for full-body fetch via handoff prompt"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
