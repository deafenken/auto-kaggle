"""Thin wrappers around the kaggle CLI, used by every stage.

Why a helper module: keeps subprocess invocation, retry/backoff, JSON-ish parsing,
and rate-limit detection in one place. The kaggle CLI is stable but its output
formats vary across commands; we normalize here so stage skills see dicts.

Usage:
    from kaggle_helpers import (
        view_competition, list_submissions, submit_csv,
        download_competition_data, list_kernels, KaggleAuthError, KaggleRateLimit,
    )

Conventions:
- All functions are idempotent and side-effect-explicit.
- All functions raise KaggleAuthError on 401/403 (caller must escalate, never retry).
- All functions retry on transient errors with exponential backoff up to 3 attempts.
- All functions log to a passed-in `progress_log_path` if provided (append-only JSONL).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class KaggleAuthError(RuntimeError):
    """401/403 from the Kaggle API. Caller must escalate; never retry."""


class KaggleRateLimit(RuntimeError):
    """429 from the Kaggle API. Caller may back off and retry."""


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_progress(progress_log_path: Optional[Path], event: dict) -> None:
    if progress_log_path is None:
        return
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts_utc": _now_utc_iso(), **event}
    with progress_log_path.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _run(cmd: list[str], timeout: int = 300) -> CmdResult:
    """Run a subprocess, returning stdout/stderr/returncode. No raise on non-zero."""
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CmdResult(proc.returncode, proc.stdout, proc.stderr)


def _classify_error(result: CmdResult) -> Optional[Exception]:
    """Look at returncode + stderr to decide retry vs escalate."""
    if result.returncode == 0:
        return None
    stderr = (result.stderr or "").lower()
    if "401" in stderr or "unauthorized" in stderr:
        return KaggleAuthError(result.stderr.strip())
    if "403" in stderr or "forbidden" in stderr:
        return KaggleAuthError(result.stderr.strip())
    if "429" in stderr or "too many requests" in stderr:
        return KaggleRateLimit(result.stderr.strip())
    return RuntimeError(
        f"kaggle command failed (code={result.returncode}): {result.stderr.strip()[:500]}"
    )


def _kaggle(
    args: list[str],
    *,
    progress_log_path: Optional[Path] = None,
    timeout: int = 300,
    max_attempts: int = 3,
) -> CmdResult:
    """Run `kaggle <args>` with auth-error escalation and rate-limit backoff."""
    cmd = ["kaggle"] + args
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        result = _run(cmd, timeout=timeout)
        err = _classify_error(result)
        if err is None:
            return result
        if isinstance(err, KaggleAuthError):
            _append_progress(
                progress_log_path,
                {"event": "kaggle_auth_error", "cmd": " ".join(shlex.quote(a) for a in cmd)},
            )
            raise err
        if isinstance(err, KaggleRateLimit):
            backoff = 60 * (2 ** (attempt - 1))
            _append_progress(
                progress_log_path,
                {"event": "rate_limited", "attempt": attempt, "backoff_seconds": backoff},
            )
            time.sleep(backoff)
            last_err = err
            continue
        last_err = err
        _append_progress(
            progress_log_path,
            {"event": "kaggle_cmd_failed", "attempt": attempt, "err": str(err)[:300]},
        )
        time.sleep(2 ** attempt)
    assert last_err is not None
    raise last_err


def view_competition(slug: str, *, progress_log_path: Optional[Path] = None) -> str:
    """Return raw stdout of `kaggle competitions view <slug>`. Caller parses."""
    result = _kaggle(["competitions", "view", slug], progress_log_path=progress_log_path)
    return result.stdout


def download_competition_data(
    slug: str, dest_dir: Path, *, progress_log_path: Optional[Path] = None
) -> int:
    """Download + unzip. Returns total bytes on disk after unzip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    _kaggle(
        ["competitions", "download", "-c", slug, "-p", str(dest_dir)],
        progress_log_path=progress_log_path,
        timeout=3600,
    )
    # Unzip any zips in dest_dir.
    for zf in dest_dir.glob("*.zip"):
        _run(["unzip", "-o", str(zf), "-d", str(dest_dir)])
        zf.unlink()
    total = sum(p.stat().st_size for p in dest_dir.rglob("*") if p.is_file())
    _append_progress(
        progress_log_path, {"stage": "stage0", "event": "data_downloaded", "bytes": total}
    )
    return total


def list_submissions(slug: str, *, progress_log_path: Optional[Path] = None) -> str:
    """Authoritative source for quota reconciliation. Returns raw stdout."""
    result = _kaggle(
        ["competitions", "submissions", "-c", slug, "-v"],
        progress_log_path=progress_log_path,
    )
    return result.stdout


def submit_csv(
    slug: str,
    csv_path: Path,
    message: str,
    *,
    progress_log_path: Optional[Path] = None,
) -> str:
    """Submit a CSV. Returns raw stdout (contains the submission ID + status)."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not message.strip():
        raise ValueError("submission message must not be empty (attribution required)")
    if "attr:" not in message and "Derived from:" not in message:
        # Integrity rule 2 — every submission message must carry attribution.
        raise ValueError(
            "submission message missing attribution. Include 'attr: <author>/<kernel>' "
            "or 'Derived from: <author>/<kernel>'."
        )
    result = _kaggle(
        ["competitions", "submit", "-c", slug, "-f", str(csv_path), "-m", message],
        progress_log_path=progress_log_path,
        timeout=900,
    )
    _append_progress(
        progress_log_path,
        {
            "stage": "stage3",
            "event": "submitted",
            "file": str(csv_path),
            "message_head": message[:80],
        },
    )
    return result.stdout


def list_kernels(
    slug: str,
    sort_by: str = "voteCount",
    page_size: int = 50,
    *,
    progress_log_path: Optional[Path] = None,
) -> str:
    """List top kernels. Returns raw stdout."""
    if sort_by not in ("voteCount", "dateCreated", "scoreAscending", "scoreDescending"):
        raise ValueError(f"unknown sort_by: {sort_by}")
    result = _kaggle(
        [
            "kernels",
            "list",
            "-c",
            slug,
            "--sort-by",
            sort_by,
            "--page-size",
            str(page_size),
            "--csv",
        ],
        progress_log_path=progress_log_path,
    )
    return result.stdout


def pull_kernel(
    author_slug: str, kernel_slug: str, dest_dir: Path, *, progress_log_path: Optional[Path] = None
) -> None:
    """Pull a kernel's source + metadata into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    _kaggle(
        ["kernels", "pull", f"{author_slug}/{kernel_slug}", "-p", str(dest_dir), "-m"],
        progress_log_path=progress_log_path,
    )
    _append_progress(
        progress_log_path,
        {
            "stage": "stage1",
            "event": "kernel_pulled",
            "author": author_slug,
            "kernel": kernel_slug,
        },
    )


# --- Quota reconciliation ----------------------------------------------------

def count_submissions_today_utc(submissions_stdout: str) -> int:
    """Parse the `kaggle competitions submissions -v` table and count today's UTC entries.

    The -v (verbose / detail) output includes timestamps in the first column. We are
    tolerant of formatting drift: any line that starts with a YYYY-MM-DD whose date
    matches today UTC counts as one submission.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    count = 0
    for line in submissions_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Look for an ISO-like date at the start.
        if len(line) >= 10 and line[:10] == today and (len(line) == 10 or not line[10].isalnum()):
            count += 1
    return count


def next_utc_midnight_iso() -> str:
    """ISO-8601 of the next UTC midnight strictly after now."""
    now = datetime.now(timezone.utc)
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if tomorrow <= now:
        # bump to next day
        from datetime import timedelta

        tomorrow = tomorrow + timedelta(days=1)
    return tomorrow.strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Heartbeat ---------------------------------------------------------------

def write_heartbeat(run_dir: Path, stage: str, substep: str, agent: str = "claude") -> None:
    """Atomic write of .heartbeat."""
    payload = {
        "stage": stage,
        "substep": substep,
        "ts_utc": _now_utc_iso(),
        "pid": os.getpid(),
        "agent": agent,
    }
    target = run_dir / ".heartbeat"
    tmp = target.with_suffix(".heartbeat.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(target)


if __name__ == "__main__":
    # Quick smoke test of the helper module — no network calls.
    print("kaggle_helpers module ok")
    print("next utc midnight:", next_utc_midnight_iso())
