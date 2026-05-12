"""Stage 1 — recon entry point.

Pulls top kernels by votes and by public LB, updates kernels_index.json, pulls
the source for new / updated kernels into runs/<slug>/stage1_recon/kernels/.

Distillation (Step 4 in SKILL.md) is the agent's job, not this script's — this
script only manages the listing + source-pull side.

Usage:
    python recon.py <comp_slug> [--runs-dir runs] [--force]

Exit codes:
    0  done (or throttled / nothing to do)
    2  bad CLI args / missing run.yaml
    3  unrecoverable error (clock skew, etc.)
    4  kaggle auth error (caller must escalate, do not retry)
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

# kaggle_helpers lives in the sibling bootstrap assets folder
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "auto-kaggle-bootstrap" / "assets"))
sys.path.insert(0, str(HERE))

from kaggle_helpers import (  # noqa: E402
    KaggleAuthError,
    KaggleRateLimit,
    list_kernels,
    pull_kernel,
    write_heartbeat,
)
from kernel_index import (  # noqa: E402
    diff_index,
    merge_listings,
    parse_kernels_csv,
    pulls_to_perform,
    write_index,
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _append_progress(progress_log: Path, event: dict[str, Any]) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts_utc": _now_iso(), **event}
    with progress_log.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _throttled(last_path: Path, interval_h: float, force: bool) -> tuple[bool, float]:
    """Return (is_throttled, age_hours)."""
    if force or not last_path.exists():
        return False, 0.0
    try:
        text = last_path.read_text().strip().replace("Z", "+00:00")
        last_ts = datetime.fromisoformat(text)
    except (ValueError, OSError):
        return False, 0.0
    age_h = (_now_utc() - last_ts).total_seconds() / 3600
    if age_h < 0:
        # Clock skew — caller escalates separately.
        return False, age_h
    return age_h < interval_h, age_h


def recon(comp_slug: str, runs_dir: Path, force: bool = False) -> int:
    run_dir = runs_dir / comp_slug
    if not run_dir.exists():
        sys.stderr.write(f"run_dir does not exist: {run_dir}\n")
        return 2

    run_yaml_path = run_dir / "run.yaml"
    if not run_yaml_path.exists():
        sys.stderr.write(f"run.yaml missing at {run_yaml_path}\n")
        return 2

    run_cfg = yaml.safe_load(run_yaml_path.read_text())
    interval_h = float(run_cfg.get("recon_interval_hours", 6))

    stage_dir = run_dir / "stage1_recon"
    progress_log = run_dir / "progress.jsonl"
    last_path = stage_dir / "last_recon_at"
    index_path = stage_dir / "kernels_index.json"
    kernels_dir = stage_dir / "kernels"
    tmp_dir = stage_dir / ".recon_tmp"

    write_heartbeat(run_dir, "stage1", "throttle_check")

    throttled, age_h = _throttled(last_path, interval_h, force)
    if age_h < 0:
        sys.stderr.write(
            f"ESCALATE: system clock is before last_recon_at by {-age_h:.2f}h — refusing\n"
        )
        return 3
    if throttled:
        print(
            f"recon throttled — last pull was {age_h:.1f}h ago, "
            f"next allowed in {interval_h - age_h:.1f}h (use --force to override)"
        )
        return 0

    write_heartbeat(run_dir, "stage1", "listing")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        by_votes_text = list_kernels(
            comp_slug, sort_by="voteCount", page_size=50, progress_log_path=progress_log
        )
        time.sleep(1.0)
        by_score_text = list_kernels(
            comp_slug, sort_by="scoreDescending", page_size=50, progress_log_path=progress_log
        )
    except KaggleAuthError as e:
        sys.stderr.write(f"ESCALATE: kaggle auth failed during listing: {e}\n")
        return 4
    except KaggleRateLimit as e:
        sys.stderr.write(f"rate-limited; bowing out for this cycle: {e}\n")
        _append_progress(progress_log, {"stage": "stage1", "event": "recon_rate_limited"})
        return 0

    _atomic_write(tmp_dir / "by_votes.csv", by_votes_text)
    _atomic_write(tmp_dir / "by_score.csv", by_score_text)

    by_votes = parse_kernels_csv(by_votes_text)
    by_score = parse_kernels_csv(by_score_text)
    _append_progress(
        progress_log,
        {
            "stage": "stage1",
            "event": "recon_listing_pulled",
            "by_votes_count": len(by_votes),
            "by_score_count": len(by_score),
        },
    )

    union = merge_listings(by_votes, by_score)
    if not union:
        print("recon: no kernels returned; nothing to do")
        _atomic_write(last_path, _now_iso())
        return 0

    # If the prior index file is corrupt, back it up before diffing.
    if index_path.exists():
        try:
            json.loads(index_path.read_text())
        except json.JSONDecodeError:
            backup = index_path.with_suffix(f".json.broken.{int(time.time())}")
            index_path.replace(backup)
            _append_progress(
                progress_log,
                {"stage": "stage1", "event": "index_recovered", "backup": str(backup)},
            )

    merged = diff_index(union, index_path)
    to_pull = pulls_to_perform(merged, cap=20)
    deferred = sum(
        1
        for e in merged.values()
        if e.status in ("new", "updated") and not e.pull_failed and e not in to_pull
    )

    pulled_ok = 0
    pulled_fail = 0
    for i, entry in enumerate(to_pull):
        write_heartbeat(run_dir, "stage1", f"pulling {entry.ref} ({i+1}/{len(to_pull)})")
        dest = kernels_dir / f"{entry.author}__{entry.slug}"
        # Clean up any half-pulled .tmp from a prior crash.
        for stale in dest.glob("*.tmp"):
            stale.unlink(missing_ok=True)
        try:
            pull_kernel(entry.author, entry.slug, dest, progress_log_path=progress_log)
            entry.last_pulled_utc = _now_iso()
            entry.pull_failed = False
            pulled_ok += 1
        except KaggleAuthError as e:
            sys.stderr.write(f"ESCALATE: kaggle auth failed mid-pull: {e}\n")
            write_index(merged, index_path)
            return 4
        except KaggleRateLimit:
            _append_progress(
                progress_log, {"stage": "stage1", "event": "recon_rate_limited", "during": "pull"}
            )
            break
        except Exception as e:  # noqa: BLE001 — best-effort, log and continue
            entry.pull_failed = True
            pulled_fail += 1
            _append_progress(
                progress_log,
                {"stage": "stage1", "event": "kernel_pull_failed", "ref": entry.ref, "err": str(e)[:200]},
            )
        if i + 1 < len(to_pull):
            time.sleep(1.0)

    if deferred:
        _append_progress(
            progress_log, {"stage": "stage1", "event": "recon_pull_capped", "deferred": deferred}
        )

    write_index(merged, index_path)
    _atomic_write(last_path, _now_iso())

    _append_progress(
        progress_log,
        {
            "stage": "stage1",
            "event": "recon_pulled",
            "kernels_total": len(merged),
            "pulled_ok": pulled_ok,
            "pulled_fail": pulled_fail,
            "deferred": deferred,
            "new": sum(1 for e in merged.values() if e.status == "new"),
            "updated": sum(1 for e in merged.values() if e.status == "updated"),
            "dropped": sum(1 for e in merged.values() if e.status == "dropped"),
        },
    )

    # Partial hand_off; the agent finishes after distillation.
    handoff_path = stage_dir / "hand_off.md"
    summary = (
        f"# Stage 1 → Stage 2 hand-off (recon, {_now_iso()})\n\n"
        f"## What I did\n"
        f"- Pulled listings (by votes: {len(by_votes)}, by public LB: {len(by_score)}).\n"
        f"- Source-pulled {pulled_ok} kernels OK, {pulled_fail} failed, {deferred} deferred to next cycle.\n"
        f"- Updated `kernels_index.json` ({len(merged)} entries total).\n\n"
        f"## What's true now\n"
        f"- New since last cycle: "
        f"{sum(1 for e in merged.values() if e.status == 'new')}\n"
        f"- Updated since last cycle: "
        f"{sum(1 for e in merged.values() if e.status == 'updated')}\n"
        f"- Dropped from top listings: "
        f"{sum(1 for e in merged.values() if e.status == 'dropped')}\n\n"
        f"## What you should do next\n"
        f"Agent: distill the pulled kernels in "
        f"`stage1_recon/kernels/<author>__<slug>/` per "
        f"`auto-kaggle-recon/references/kernel-distillation.md`, write entries to "
        f"`ideas_pool.md` per `auto-kaggle-recon/references/ideas-pool-format.md`, "
        f"and add citations to `citations.bib`. Skip kernels whose "
        f"`distilled_at_utc` is already set. Update the Consensus / New-this-cycle "
        f"summary at the top of `ideas_pool.md` when done.\n"
    )
    _atomic_write(handoff_path, summary)

    print(
        f"recon ok: {pulled_ok} pulled, {pulled_fail} failed, {deferred} deferred, "
        f"{len(merged)} total in index — agent must distill"
    )
    write_heartbeat(run_dir, "stage1", "pull_done_pending_distillation")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("comp_slug")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--force", action="store_true", help="bypass the throttle window")
    args = ap.parse_args()
    return recon(args.comp_slug, Path(args.runs_dir), force=args.force)


if __name__ == "__main__":
    sys.exit(main())
