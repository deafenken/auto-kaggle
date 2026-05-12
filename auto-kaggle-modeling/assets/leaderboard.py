"""Append-only manager for stage2_modeling/leaderboard.csv.

The leaderboard is the rolling table of every completed run. Stage 3 reads it
to rank submission candidates. Schema:

    run_id, ts_utc, template, idea_keys, cv_metric, cv_score, cv_std,
    public_lb, gap, status, attribution_keys

- `idea_keys` and `attribution_keys` are pipe-separated within the CSV cell so
  we don't fight CSV quoting; both also live in the per-run attribution.md as
  the authoritative source. The cell here is a convenience for fast reads.
- `status`: one of {running, completed, terminated_overbudget, failed}
- `public_lb` and `gap` start as null; Stage 3 fills them after a submission lands.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LEADERBOARD_HEADER = [
    "run_id",
    "ts_utc",
    "template",
    "idea_keys",
    "cv_metric",
    "cv_score",
    "cv_std",
    "public_lb",
    "gap",
    "status",
    "attribution_keys",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LEADERBOARD_HEADER})
    tmp.replace(path)


def load(leaderboard_path: Path) -> list[dict]:
    if not leaderboard_path.exists():
        return []
    with leaderboard_path.open() as f:
        rows = list(csv.DictReader(f))
    return rows


def append_or_update(leaderboard_path: Path, row: dict) -> None:
    """Insert or update by run_id. Atomic rewrite (entire file is small).

    Order: by ts_utc descending so most recent runs are on top. Ties broken by
    cv_score in the direction the metric prefers (we don't enforce here — the
    reader sorts as it pleases).
    """
    rows = load(leaderboard_path)
    by_id = {r["run_id"]: r for r in rows}
    by_id[row["run_id"]] = {**by_id.get(row["run_id"], {}), **row}
    rows = list(by_id.values())
    rows.sort(key=lambda r: r.get("ts_utc", ""), reverse=True)
    _atomic_write_csv(leaderboard_path, rows)


def record_run_start(
    leaderboard_path: Path, run_id: str, template: str, idea_keys: list[str]
) -> None:
    append_or_update(
        leaderboard_path,
        {
            "run_id": run_id,
            "ts_utc": _now_iso(),
            "template": template,
            "idea_keys": "|".join(idea_keys),
            "status": "running",
        },
    )


def record_run_completed(
    leaderboard_path: Path,
    run_id: str,
    cv_metric: str,
    cv_score: float,
    cv_std: float,
    attribution_keys: list[str],
    template: Optional[str] = None,
    idea_keys: Optional[list[str]] = None,
) -> None:
    row: dict = {
        "run_id": run_id,
        "ts_utc": _now_iso(),
        "cv_metric": cv_metric,
        "cv_score": f"{cv_score:.6f}",
        "cv_std": f"{cv_std:.6f}",
        "status": "completed",
        "attribution_keys": "|".join(attribution_keys),
    }
    if template is not None:
        row["template"] = template
    if idea_keys is not None:
        row["idea_keys"] = "|".join(idea_keys)
    append_or_update(leaderboard_path, row)


def record_run_failed(
    leaderboard_path: Path, run_id: str, reason: str = "failed", err: str = ""
) -> None:
    append_or_update(
        leaderboard_path,
        {"run_id": run_id, "ts_utc": _now_iso(), "status": reason},
    )
    # Also write a sibling .err file with the error detail (not into the CSV).
    err_path = leaderboard_path.with_name(f"{run_id}.err")
    err_path.write_text(err)


def record_submission_result(
    leaderboard_path: Path, run_id: str, public_lb: float, cv_score: Optional[float] = None
) -> None:
    """Called by Stage 3 after a submission's public LB is known."""
    row = {"run_id": run_id, "public_lb": f"{public_lb:.6f}"}
    if cv_score is not None:
        row["gap"] = f"{abs(public_lb - cv_score):.6f}"
    append_or_update(leaderboard_path, row)


def best_by_cv(
    leaderboard_path: Path, direction: str = "max", trustworthy_only: bool = True
) -> Optional[dict]:
    """Return the best completed row by cv_score in `direction` ('max' or 'min').

    `trustworthy_only`: drop rows where cv_std == "" or cv_std > 0.5 * |cv_score|
    (a crude trust filter; Stage 3 has the proper trust-adjusted ranking).
    """
    rows = [r for r in load(leaderboard_path) if r.get("status") == "completed" and r.get("cv_score")]
    if trustworthy_only:
        kept = []
        for r in rows:
            try:
                cs = float(r["cv_score"])
                std = float(r.get("cv_std") or 0)
            except ValueError:
                continue
            if cs == 0:
                continue
            if std and std > 0.5 * abs(cs):
                continue
            kept.append(r)
        rows = kept
    if not rows:
        return None
    key = (lambda r: float(r["cv_score"])) if direction == "max" else (lambda r: -float(r["cv_score"]))
    return max(rows, key=key)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("leaderboard_path")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    rows = load(Path(args.leaderboard_path))
    for r in rows[: args.top]:
        print(json.dumps(r))
