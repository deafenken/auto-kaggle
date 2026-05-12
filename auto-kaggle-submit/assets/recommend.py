"""Ranking + recommendations.md writer.

Reads leaderboard.csv + submission_log.jsonl + each run's attribution.md and
produces a ranked recommendations.md per the rubric in
auto-kaggle-submit/references/ranking-rubric.md.

Does NOT submit anything. Submission is in submit.py.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from near_duplicate import load_predictions, mae  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_leaderboard(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _read_submission_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _trust_adjusted(cv: float, std: float, direction: str, alpha: float) -> float:
    if direction == "maximize":
        return cv - alpha * std
    return cv + alpha * std


def _last_public_lb_for_run(run_id: str, submissions: list[dict]) -> Optional[float]:
    for entry in reversed(submissions):
        if entry.get("run_id") == run_id and entry.get("public_lb") is not None:
            try:
                return float(entry["public_lb"])
            except (TypeError, ValueError):
                continue
    return None


def _disagreement_with(
    candidate_preds_path: Path, prior_preds_path: Path
) -> Optional[float]:
    try:
        a = load_predictions(candidate_preds_path)
        b = load_predictions(prior_preds_path)
        if a.shape != b.shape:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:  # noqa: BLE001
        return None


def _read_attribution_keys(attr_path: Path) -> list[str]:
    """Pull `attr: <author>/<kernel>, ...` style citations from attribution.md."""
    if not attr_path.exists():
        return []
    text = attr_path.read_text()
    keys: list[str] = []
    in_citations = False
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("## citations"):
            in_citations = True
            continue
        if in_citations and s.startswith("##"):
            break
        if in_citations and s.startswith("- "):
            # "- jdoe123/eda-and-lgbm-baseline (Apache-2.0)"
            ref = s[2:].split()[0].strip(",")
            if "/" in ref:
                keys.append(ref)
    return keys


def calibration_health(submissions: list[dict], default_alpha: float = 1.0) -> str:
    """Look at last 3 submissions with public_lb; classify gap vs cv_std."""
    relevant = [
        s for s in submissions
        if s.get("public_lb") is not None and s.get("cv_score") is not None and s.get("cv_std")
    ][-3:]
    if not relevant:
        return "unknown"
    breaches = 0
    severe = 0
    for s in relevant:
        try:
            gap = abs(float(s["public_lb"]) - float(s["cv_score"]))
            std = float(s["cv_std"])
            if std == 0:
                continue
            ratio = gap / std
            if ratio > 2:
                severe += 1
            elif ratio > 1:
                breaches += 1
        except (TypeError, ValueError):
            continue
    if severe:
        return "suspect"
    if breaches >= 2:
        return "mild"
    return "healthy"


def build_recommendations(
    run_dir: Path,
    leaderboard: list[dict],
    submissions: list[dict],
    quota_line: str,
    metric_direction: str,
    alpha: float,
    metric_name: str,
    deadline_utc: Optional[str],
) -> str:
    """Build the recommendations.md text."""
    completed = [
        r for r in leaderboard
        if r.get("status") == "completed" and r.get("cv_score") not in (None, "")
    ]
    # Rank entries
    ranked: list[dict] = []
    for r in completed:
        try:
            cv = float(r["cv_score"])
            std = float(r.get("cv_std") or 0)
        except ValueError:
            continue
        r2 = dict(r)
        r2["_cv"] = cv
        r2["_std"] = std
        r2["_trust"] = _trust_adjusted(cv, std, metric_direction, alpha)
        r2["_untrustworthy"] = (std > 0.5 * abs(cv)) if cv != 0 else False
        r2["_public_lb"] = _last_public_lb_for_run(r["run_id"], submissions)
        # Near-duplicate check
        cand_preds = run_dir / "stage2_modeling" / "runs" / r["run_id"] / "test_preds.csv"
        r2["_near_dup_of"] = None
        if cand_preds.exists():
            for s in submissions:
                prior = Path(s.get("file") or "")
                if not prior.exists():
                    continue
                try:
                    d = mae(load_predictions(cand_preds), load_predictions(prior))
                    if d < 1e-6:
                        r2["_near_dup_of"] = s.get("run_id")
                        break
                except Exception:  # noqa: BLE001
                    continue
        # Attribution count for tiebreaks / display
        attr_path = run_dir / "stage2_modeling" / "runs" / r["run_id"] / "attribution.md"
        r2["_attr_keys"] = _read_attribution_keys(attr_path)
        ranked.append(r2)

    # Sort by trust_adjusted (preferred direction)
    reverse = metric_direction == "maximize"
    ranked.sort(key=lambda r: r["_trust"], reverse=reverse)

    # Slot selections
    eligible = [r for r in ranked if not r["_untrustworthy"] and not r["_near_dup_of"]]
    recommended = eligible[0] if eligible else None
    safe = None
    if eligible:
        top_n = eligible[: max(5, len(eligible))][:5]
        safe = min(top_n, key=lambda r: r["_std"])
        if recommended is not None and safe["run_id"] == recommended["run_id"]:
            # If they coincide, pick a different SAFE
            rest = [r for r in top_n if r["run_id"] != recommended["run_id"]]
            safe = min(rest, key=lambda r: r["_std"]) if rest else None

    health = calibration_health(submissions)

    # Build text
    lines = [
        f"# Recommendations — refreshed {_now_iso()}",
        "",
        f"_{quota_line}_",
        f"_CV-LB calibration: {health}._",
        f"_Metric: {metric_name} ({metric_direction})._",
    ]
    if deadline_utc:
        lines.append(f"_Deadline: {deadline_utc} (UTC)._")
    lines.append("")

    def _entry_block(label: str, r: dict) -> list[str]:
        public_lb = r["_public_lb"]
        gap_text = (
            f" · gap {abs(public_lb - r['_cv']):.4f}"
            if public_lb is not None
            else " · gap n/a"
        )
        return [
            f"## {label} — {r['run_id']}",
            "",
            f"- **CV ({metric_name}):** {r['_cv']:.6f} (std {r['_std']:.6f})  ·  "
            f"**Trust-adjusted:** {r['_trust']:.6f}",
            f"- **Public LB (last):** "
            + (f"{public_lb:.6f}{gap_text}" if public_lb is not None else "n/a"),
            f"- **Ideas:** {r.get('idea_keys', '') or '(none recorded)'}",
            f"- **Attribution:** "
            + (", ".join(r["_attr_keys"]) if r["_attr_keys"] else "(none — Rule 2 violation, will not submit)"),
            "",
        ]

    if recommended is not None:
        lines.extend(_entry_block("RECOMMENDED", recommended))
    else:
        lines += [
            "## RECOMMENDED — (none available)",
            "",
            "No completed run passes the trust filter. Need either a new run or to relax `trust_alpha`.",
            "",
        ]

    if safe is not None:
        lines.extend(_entry_block("SAFE", safe))

    # Other ranked candidates (deduped vs the two slots)
    shown_ids = {x["run_id"] for x in (recommended, safe) if x is not None}
    others = [r for r in ranked if r["run_id"] not in shown_ids]
    if others:
        lines += [
            "## Other ranked candidates",
            "",
            "| Rank | run_id | CV | std | Trust-adj | Public LB | Status | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for i, r in enumerate(others, start=3 if safe else 2):
            notes = []
            if r["_untrustworthy"]:
                notes.append(f"untrustworthy (std/cv={r['_std'] / max(abs(r['_cv']), 1e-9):.2f})")
            if r["_near_dup_of"]:
                notes.append(f"near_duplicate of {r['_near_dup_of']}")
            rank_marker = "⚠" if (r["_untrustworthy"] or r["_near_dup_of"]) else str(i)
            lb_text = f"{r['_public_lb']:.6f}" if r["_public_lb"] is not None else "n/a"
            lines.append(
                f"| {rank_marker} | {r['run_id']} | {r['_cv']:.4f} | {r['_std']:.4f} | "
                f"{r['_trust']:.4f} | {lb_text} | completed | {'; '.join(notes) or '—'} |"
            )
        lines.append("")

    lines += [
        "## How to submit one",
        "",
        "Tell me in your next message:",
        "- `submit candidate 1`  (or `submit <run_id>`)",
        "- `submit candidate 2 with message: <text>`",
        "- `quota only`  (just refresh, no submit)",
        "",
        "Reminder: the final 2 submissions before deadline are always user-picked,",
        "never auto-submitted (rules 3 and 9). A separate `final_selection.md` appears",
        "in the last 24h before deadline.",
        "",
    ]
    return "\n".join(lines)


def write_recommendations(run_dir: Path, text: str) -> Path:
    out_path = run_dir / "stage3_submit" / "recommendations.md"
    tmp = out_path.with_suffix(".md.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text)
    tmp.replace(out_path)
    return out_path


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("comp_slug")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--quota-line", default="")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    run_dir = Path(args.runs_dir) / args.comp_slug
    cp_path = run_dir / "stage0_bootstrap" / "comp_profile.yaml"
    cp = yaml.safe_load(cp_path.read_text())
    metric = cp.get("metric") or {}
    leaderboard = _read_leaderboard(run_dir / "stage2_modeling" / "leaderboard.csv")
    submissions = _read_submission_log(run_dir / "stage3_submit" / "submission_log.jsonl")
    text = build_recommendations(
        run_dir,
        leaderboard,
        submissions,
        quota_line=args.quota_line,
        metric_direction=(metric.get("direction") or "maximize"),
        alpha=args.alpha,
        metric_name=(metric.get("name") or "metric"),
        deadline_utc=cp.get("deadline_utc"),
    )
    out = write_recommendations(run_dir, text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
