"""Parse `kaggle kernels list --csv` output and maintain kernels_index.json.

The kaggle CLI's --csv output has these columns (as of the kaggle package
versions in active use): `ref,title,author,lastRunTime,totalVotes`. We parse
defensively (column order may shift in future versions; missing fields tolerated).

kernels_index.json schema (one object per kernel):

    {
      "ref": "<author>/<slug>",
      "author": "<author>",
      "slug": "<slug>",
      "title": "<title>",
      "votes": <int>,
      "public_lb": <float | null>,
      "last_run_time_utc": "<iso8601>",
      "last_pulled_utc": "<iso8601 | null>",
      "distilled_at_utc": "<iso8601 | null>",
      "in_top_votes": <bool>,
      "in_top_score": <bool>,
      "status": "new | updated | unchanged | dropped",
      "dropped_at_utc": "<iso8601 | null>",
      "pull_failed": <bool>,
      "low_signal": <bool>,
      "off_topic": <bool>,
      "rule_violation_suspect": <bool>
    }
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class KernelEntry:
    ref: str
    author: str
    slug: str
    title: str
    votes: int = 0
    public_lb: Optional[float] = None
    last_run_time_utc: Optional[str] = None
    last_pulled_utc: Optional[str] = None
    distilled_at_utc: Optional[str] = None
    in_top_votes: bool = False
    in_top_score: bool = False
    status: str = "new"
    dropped_at_utc: Optional[str] = None
    pull_failed: bool = False
    low_signal: bool = False
    off_topic: bool = False
    rule_violation_suspect: bool = False


def parse_kernels_csv(text: str) -> list[KernelEntry]:
    """Parse `kaggle kernels list --csv` output.

    Tolerates column reordering by looking up by header name. Skips rows that
    cannot be parsed. Returns an empty list if the CSV is empty / malformed.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    reader = csv.DictReader(lines)
    out: list[KernelEntry] = []
    for row in reader:
        ref = (row.get("ref") or "").strip()
        if not ref or "/" not in ref:
            continue
        author, slug = ref.split("/", 1)
        try:
            votes = int(row.get("totalVotes") or 0)
        except (ValueError, TypeError):
            votes = 0
        # public_lb is not always present in kernels list output; may live in a
        # `scoreData` column for some kaggle CLI versions. We accept missing.
        public_lb_raw = row.get("publicScore") or row.get("score") or ""
        try:
            public_lb: Optional[float] = float(public_lb_raw) if public_lb_raw else None
        except ValueError:
            public_lb = None
        out.append(
            KernelEntry(
                ref=ref,
                author=author,
                slug=slug,
                title=(row.get("title") or "").strip() or ref,
                votes=votes,
                public_lb=public_lb,
                last_run_time_utc=(row.get("lastRunTime") or "").strip() or None,
            )
        )
    return out


def merge_listings(
    by_votes: list[KernelEntry], by_score: list[KernelEntry]
) -> dict[str, KernelEntry]:
    """Union the two listings by `ref`, preserving membership flags."""
    merged: dict[str, KernelEntry] = {}
    for k in by_votes:
        k.in_top_votes = True
        merged[k.ref] = k
    for k in by_score:
        if k.ref in merged:
            merged[k.ref].in_top_score = True
            # Prefer the higher vote count if listings disagree.
            if k.votes > merged[k.ref].votes:
                merged[k.ref].votes = k.votes
            if k.public_lb is not None:
                merged[k.ref].public_lb = k.public_lb
        else:
            k.in_top_score = True
            merged[k.ref] = k
    return merged


def diff_index(
    new_union: dict[str, KernelEntry], prev_index_path: Path
) -> dict[str, KernelEntry]:
    """Classify each kernel as new/updated/unchanged/dropped against the prior index.

    Returns a merged index where existing entries retain their `last_pulled_utc`,
    `distilled_at_utc`, and qualitative flags.
    """
    prev: dict[str, dict] = {}
    if prev_index_path.exists():
        try:
            data = json.loads(prev_index_path.read_text())
            for entry in data:
                prev[entry["ref"]] = entry
        except (json.JSONDecodeError, KeyError):
            # Corrupt or partial index — caller is expected to back this up
            # before invoking us (recon SKILL.md "kernels_index.json broken" case).
            prev = {}

    out: dict[str, KernelEntry] = {}
    now = _now_utc_iso()

    for ref, fresh in new_union.items():
        if ref not in prev:
            fresh.status = "new"
            out[ref] = fresh
            continue
        old = prev[ref]
        # Preserve durable fields.
        fresh.last_pulled_utc = old.get("last_pulled_utc")
        fresh.distilled_at_utc = old.get("distilled_at_utc")
        fresh.low_signal = bool(old.get("low_signal", False))
        fresh.off_topic = bool(old.get("off_topic", False))
        fresh.rule_violation_suspect = bool(old.get("rule_violation_suspect", False))
        fresh.pull_failed = bool(old.get("pull_failed", False))
        # Determine updated vs unchanged.
        if (
            fresh.last_run_time_utc
            and old.get("last_pulled_utc")
            and fresh.last_run_time_utc > old["last_pulled_utc"]
        ):
            fresh.status = "updated"
        elif old.get("last_pulled_utc") is None:
            fresh.status = "new"  # pulled-never-finished gets retried
        else:
            fresh.status = "unchanged"
        out[ref] = fresh

    # Mark dropped entries (in prev but not in fresh).
    for ref, old in prev.items():
        if ref in out:
            continue
        entry = KernelEntry(
            ref=ref,
            author=old.get("author", ref.split("/", 1)[0]),
            slug=old.get("slug", ref.split("/", 1)[1] if "/" in ref else ref),
            title=old.get("title", ref),
            votes=int(old.get("votes", 0)),
            public_lb=old.get("public_lb"),
            last_run_time_utc=old.get("last_run_time_utc"),
            last_pulled_utc=old.get("last_pulled_utc"),
            distilled_at_utc=old.get("distilled_at_utc"),
            in_top_votes=False,
            in_top_score=False,
            status="dropped",
            dropped_at_utc=old.get("dropped_at_utc") or now,
            pull_failed=bool(old.get("pull_failed", False)),
            low_signal=bool(old.get("low_signal", False)),
            off_topic=bool(old.get("off_topic", False)),
            rule_violation_suspect=bool(old.get("rule_violation_suspect", False)),
        )
        out[ref] = entry

    return out


def write_index(index: dict[str, KernelEntry], path: Path) -> None:
    """Atomic write of the index, sorted by votes desc then ref asc for stable diffs."""
    entries = [asdict(v) for v in index.values()]
    entries.sort(key=lambda e: (-int(e.get("votes") or 0), e["ref"]))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    tmp.replace(path)


def pulls_to_perform(index: dict[str, KernelEntry], cap: int = 20) -> list[KernelEntry]:
    """Return kernels with status in {new, updated} prioritized for this cycle.

    Priority score: `votes + 100 * (1 if public_lb is not None else 0)`. The boost
    for has-public-LB ensures freshly-published high-LB kernels are pulled even if
    they have no votes yet.
    """
    candidates = [e for e in index.values() if e.status in ("new", "updated") and not e.pull_failed]
    candidates.sort(key=lambda e: -(int(e.votes) + (100 if e.public_lb is not None else 0)))
    return candidates[:cap]
