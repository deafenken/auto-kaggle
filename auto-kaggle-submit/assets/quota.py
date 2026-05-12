"""Quota reconciliation + wait_until protocol for Stage 3.

The local quota_state.yaml is a cache. `kaggle competitions submissions -c <slug>`
is the truth. This module:

  - parses Kaggle's text output
  - counts today's submissions in UTC
  - writes quota_state.yaml atomically
  - writes / reads wait_until.txt
  - detects clock skew

See auto-kaggle-submit/references/quota-reconciliation.md for the why.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "auto-kaggle-bootstrap" / "assets"))
from kaggle_helpers import KaggleAuthError, list_submissions  # noqa: E402


@dataclass
class QuotaState:
    used_today: int
    daily_limit: int
    remaining: int
    last_submission_utc: Optional[str]
    last_reset_utc: str
    next_reset_utc: str
    exhausted: bool


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc_today_midnight() -> datetime:
    return _now_utc().replace(hour=0, minute=0, second=0, microsecond=0)


def _utc_tomorrow_midnight() -> datetime:
    return _utc_today_midnight() + timedelta(days=1)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# `kaggle competitions submissions -v` produces a text table whose data rows
# start with a `YYYY-MM-DD HH:MM:SS` timestamp. We detect those defensively.
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]\d{2}:\d{2}:\d{2}")


def parse_submissions_text(text: str, today_utc_date: str) -> tuple[int, Optional[str]]:
    """Return (count of today's submissions, last submission ISO timestamp)."""
    used = 0
    last_ts: Optional[str] = None
    for line in text.splitlines():
        m = _TS_RE.match(line.strip())
        if not m:
            continue
        date = m.group(1)
        # Reconstruct ISO timestamp.
        # The original tokens look like "2026-05-12 14:22:03"; transform to ISO.
        ts_str = line.strip()[: 19].replace(" ", "T") + "Z"
        if last_ts is None or ts_str > last_ts:
            last_ts = ts_str
        if date == today_utc_date:
            used += 1
    return used, last_ts


def load_quota_state(path: Path) -> Optional[QuotaState]:
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
        return QuotaState(**{k: data.get(k) for k in QuotaState.__annotations__})
    except (yaml.YAMLError, TypeError):
        return None


def write_quota_state(path: Path, state: QuotaState) -> None:
    _atomic_write_text(path, yaml.safe_dump(asdict(state), sort_keys=False))


def write_wait_until(stage_dir: Path, ts_utc: str) -> None:
    _atomic_write_text(stage_dir / "wait_until.txt", ts_utc + ("\n" if not ts_utc.endswith("\n") else ""))


def read_wait_until(stage_dir: Path) -> Optional[datetime]:
    p = stage_dir / "wait_until.txt"
    if not p.exists():
        return None
    raw = p.read_text().strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def clear_wait_until(stage_dir: Path) -> None:
    p = stage_dir / "wait_until.txt"
    if p.exists():
        p.unlink()


def fmt_until(target: datetime) -> str:
    delta = target - _now_utc()
    total_s = max(int(delta.total_seconds()), 0)
    h = total_s // 3600
    m = (total_s % 3600) // 60
    return f"{h}h {m:02d}m"


def reconcile(
    comp_slug: str,
    stage_dir: Path,
    daily_limit: int,
    progress_log: Optional[Path] = None,
) -> tuple[QuotaState, str]:
    """Pull Kaggle's truth, update the cache, return state + a human-readable status line.

    Raises KaggleAuthError on auth failures (caller escalates).
    """
    raw = list_submissions(comp_slug, progress_log_path=progress_log)
    today_str = _utc_today_midnight().strftime("%Y-%m-%d")
    used_today, last_ts = parse_submissions_text(raw, today_str)

    cached = load_quota_state(stage_dir / "quota_state.yaml")
    if cached is not None and abs((cached.used_today or 0) - used_today) > 1:
        sys.stderr.write(
            f"ESCALATE: quota desync — Kaggle reports {used_today} today, "
            f"local cache says {cached.used_today}. Taking Kaggle as truth.\n"
        )
        # Caller still proceeds, but with the escalation flag in progress_log.
        if progress_log is not None:
            payload = {
                "ts_utc": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "stage": "stage3",
                "event": "quota_desync",
                "kaggle": used_today,
                "local": cached.used_today,
            }
            with progress_log.open("a") as f:
                f.write(json.dumps(payload) + "\n")

    state = QuotaState(
        used_today=used_today,
        daily_limit=daily_limit,
        remaining=max(0, daily_limit - used_today),
        last_submission_utc=last_ts,
        last_reset_utc=_utc_today_midnight().strftime("%Y-%m-%dT%H:%M:%SZ"),
        next_reset_utc=_utc_tomorrow_midnight().strftime("%Y-%m-%dT%H:%M:%SZ"),
        exhausted=(used_today >= daily_limit),
    )
    write_quota_state(stage_dir / "quota_state.yaml", state)

    status_line = (
        f"Quota {state.used_today}/{state.daily_limit}  "
        f"Remaining {state.remaining}  "
        f"Next reset in {fmt_until(_utc_tomorrow_midnight())} ({state.next_reset_utc})"
    )
    return state, status_line


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("comp_slug")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--daily-limit", type=int, default=5)
    args = ap.parse_args()
    stage_dir = Path(args.runs_dir) / args.comp_slug / "stage3_submit"
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        state, line = reconcile(args.comp_slug, stage_dir, args.daily_limit)
    except KaggleAuthError as e:
        sys.stderr.write(f"ESCALATE auth: {e}\n")
        return 4
    print(f"[auto-kaggle-submit] {args.comp_slug}  {line}")
    if state.exhausted:
        wu = state.next_reset_utc
        write_wait_until(stage_dir, wu)
        print(f"WAIT_UNTIL {wu} (in {fmt_until(_utc_tomorrow_midnight())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
