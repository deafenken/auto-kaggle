"""Near-duplicate detection for submission predictions.

Used by Stage 3 to enforce Rule 4 — no LB probing via near-identical submissions.

Two prediction files are "near-duplicates" if their mean absolute element-wise
difference is below a threshold (default 1e-6).

Two cases this catches:
  1. Honest mistake: submitting the same run_id twice (the test_preds didn't change).
  2. Probing: deliberately tweaking one row at a time to extract test-set signal.

In both cases the local block is the right response. The user can override with
--allow-near-duplicate + a written reason — that's the audit trail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_predictions(csv_path: Path, id_col: Optional[str] = None) -> np.ndarray:
    """Load a submission CSV as a flat float array (id-aligned, id col dropped).

    The id column is auto-detected if not provided (first column, case-insensitive
    match on "id" / "image_id" / "sample_id" / "series_id" / first column).
    """
    df = pd.read_csv(csv_path)
    if id_col is None:
        for cand in ("id", "image_id", "sample_id", "series_id"):
            if cand in df.columns:
                id_col = cand
                break
        if id_col is None:
            id_col = df.columns[0]
    df = df.sort_values(id_col).reset_index(drop=True)
    val_cols = [c for c in df.columns if c != id_col]
    arr = df[val_cols].to_numpy(dtype=np.float64).ravel()
    return arr


def mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.mean(np.abs(a - b)))


def check_against_log(
    candidate_csv: Path,
    submission_log_path: Path,
    threshold: float = 1e-6,
    id_col: Optional[str] = None,
) -> tuple[bool, float, Optional[str]]:
    """Compare candidate's predictions to every entry in submission_log.jsonl.

    Returns (is_near_duplicate, min_mae, matching_run_id).
    """
    if not submission_log_path.exists():
        return False, float("inf"), None
    candidate = load_predictions(candidate_csv, id_col=id_col)
    min_mae = float("inf")
    matching_run_id: Optional[str] = None
    with submission_log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            prior_path = Path(entry.get("file", ""))
            if not prior_path.exists():
                continue
            try:
                prior = load_predictions(prior_path, id_col=id_col)
            except Exception:  # noqa: BLE001
                continue
            d = mae(candidate, prior)
            if d < min_mae:
                min_mae = d
                matching_run_id = entry.get("run_id")
    return (min_mae < threshold), min_mae, matching_run_id


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_csv")
    ap.add_argument("submission_log_jsonl")
    ap.add_argument("--threshold", type=float, default=1e-6)
    ap.add_argument("--id-col", default=None)
    args = ap.parse_args()
    is_dup, d, match = check_against_log(
        Path(args.candidate_csv),
        Path(args.submission_log_jsonl),
        threshold=args.threshold,
        id_col=args.id_col,
    )
    print(json.dumps({"near_duplicate": is_dup, "min_mae": d, "matches": match}))
    return 0 if not is_dup else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
