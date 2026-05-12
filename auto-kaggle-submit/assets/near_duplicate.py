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


def load_predictions(
    csv_path: Path,
    id_col: Optional[str] = None,
    prediction_cols: Optional[list[str]] = None,
) -> np.ndarray:
    """Load a submission CSV as a flat float array (id-aligned, id col dropped).

    - `id_col`: the column to sort by (so rows align across files). Auto-detected
      from common names if not supplied.
    - `prediction_cols`: numeric columns to compare on. If supplied, only these
      are loaded as floats. If None, all non-id columns are tried; non-numeric
      ones (event labels, RLE strings, prediction strings) are encoded by their
      hash mod 2^53 so a near-duplicate check still works on string-valued
      submissions without crashing.

    Call sites that have a `comp_profile.submission.columns` list SHOULD pass
    `prediction_cols` explicitly — that way the comparison is exact and
    detection of LB probing remains tight.
    """
    df = pd.read_csv(csv_path)
    if id_col is None:
        for cand in ("id", "image_id", "sample_id", "series_id", "Id", "ImageId"):
            if cand in df.columns:
                id_col = cand
                break
        if id_col is None:
            id_col = df.columns[0]
    df = df.sort_values(id_col).reset_index(drop=True)

    cols = prediction_cols if prediction_cols else [c for c in df.columns if c != id_col]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        # Nothing to compare on — pretend single 0 value per row.
        return np.zeros(len(df), dtype=np.float64)

    pieces: list[np.ndarray] = []
    for c in cols:
        s = df[c]
        if s.dtype.kind in "fiub":
            pieces.append(s.to_numpy(dtype=np.float64))
            continue
        # Non-numeric column. Try float coercion first; if it fails, hash.
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().all():
            pieces.append(coerced.to_numpy(dtype=np.float64))
            continue
        # Stable hash mod 2^53 keeps values within float64 exact-int range so
        # MAE between two identical string-valued columns is exactly 0.
        pieces.append(
            s.fillna("").astype(str).map(lambda v: float(hash(v) % (2 ** 53))).to_numpy(dtype=np.float64)
        )
    arr = np.stack(pieces, axis=-1).ravel()
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
    prediction_cols: Optional[list[str]] = None,
) -> tuple[bool, float, Optional[str]]:
    """Compare candidate's predictions to every entry in submission_log.jsonl.

    Returns (is_near_duplicate, min_mae, matching_run_id). Skips follow-up
    records (`public_lb_known`) that have no `file` field. Pass `prediction_cols`
    from `comp_profile.submission.columns` for exact targeting.
    """
    if not submission_log_path.exists():
        return False, float("inf"), None
    candidate = load_predictions(candidate_csv, id_col=id_col, prediction_cols=prediction_cols)
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
            file_field = entry.get("file")
            if not file_field:
                continue   # e.g. public_lb_known follow-ups have no file
            prior_path = Path(file_field)
            if not prior_path.exists():
                continue
            try:
                prior = load_predictions(prior_path, id_col=id_col, prediction_cols=prediction_cols)
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
