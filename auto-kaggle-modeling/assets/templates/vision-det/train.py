"""Detection training template — SKELETON, agent must implement framework-specific logic.

Detection is too framework-dependent for a single skeleton to cover both training
and inference end-to-end. This file provides:

  - The same outer contract as other templates (config / progress events /
    cv_score.json / test_preds.csv / atomic fold checkpoints).
  - Three framework hooks: `train_one_fold_ultralytics`, `train_one_fold_mmdet`,
    `train_one_fold_torchvision` — agent picks one based on cfg.framework and
    customizes the picked one.
  - Loaders that translate the competition's COCO-style or per-image annotation
    files into the chosen framework's expected format.

CV in detection is harder than classification because validation needs IoU + mAP
on bounding boxes, not a simple scalar per row. We compute COCO-style mAP per
fold via pycocotools (or the framework's built-in evaluator). The agent must
implement annotation conversion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_progress(progress_log: Path, event: dict[str, Any]) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts_utc": _now_iso(), **event}
    with progress_log.open("a") as f:
        f.write(json.dumps(event) + "\n")


# ---------------------------------------------------------------------------
# Functions the agent MUST implement.
# ---------------------------------------------------------------------------


def convert_annotations_for_framework(cfg: dict[str, Any], train_df: pd.DataFrame, fold_indices):
    """Write per-fold annotation files in the format the chosen framework wants.

    - For ultralytics: a YOLO-format `data.yaml` + per-image .txt labels in a
      framework-specific folder layout.
    - For mmdet: a COCO-format JSON per fold (train + val).
    - For torchvision: a `CocoDetection`-compatible JSON.

    Return the paths the trainer needs.
    """
    raise NotImplementedError(
        "agent must implement convert_annotations_for_framework() — this is "
        "where the competition's annotation format gets translated."
    )


def train_one_fold_ultralytics(cfg, paths) -> tuple[float, np.ndarray]:
    """Train ultralytics YOLO for one fold. Return (mAP score, test predictions array)."""
    raise NotImplementedError(
        "agent must implement train_one_fold_ultralytics() using "
        "`ultralytics.YOLO(cfg['model_name']).train(data=paths.data_yaml, "
        "epochs=cfg['epochs'], imgsz=cfg['image_size'], batch=cfg['batch_size'])`. "
        "Capture val mAP from results, run prediction on test set, return."
    )


def train_one_fold_mmdet(cfg, paths) -> tuple[float, np.ndarray]:
    raise NotImplementedError("agent must implement train_one_fold_mmdet() using mmdet's Runner")


def train_one_fold_torchvision(cfg, paths) -> tuple[float, np.ndarray]:
    raise NotImplementedError("agent must implement train_one_fold_torchvision()")


def encode_submission(cfg, test_predictions) -> pd.DataFrame:
    """Detection submissions are usually one row per (image, predicted_box) tuple
    with confidence. Comp-specific encoding (COCO JSON inside a column, or
    `PredictionString` per image, etc.)."""
    raise NotImplementedError("agent must implement encode_submission()")


# ---------------------------------------------------------------------------
# Outer loop.
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--progress-log", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    progress_log = Path(args.progress_log)
    cfg = yaml.safe_load((run_dir / args.config).read_text())

    fw = cfg.get("framework", "ultralytics")
    train_one_fold = {
        "ultralytics": train_one_fold_ultralytics,
        "mmdet": train_one_fold_mmdet,
        "torchvision": train_one_fold_torchvision,
    }.get(fw)
    if train_one_fold is None:
        sys.stderr.write(f"unknown framework: {fw}\n")
        return 2

    seeds = cfg.get("seeds", [42])
    n_folds = int(cfg["cv_split"]["n_folds"])
    folds_dir = run_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(cfg["data_paths"]["train"])
    y_proxy = train_df.index.to_numpy()

    sys.path.insert(0, str(Path(__file__).parent.parent / "tabular-lgbm"))
    from train import _split_iter  # noqa: PLC0415

    per_fold_scores: list[float] = []
    test_preds_all: list[np.ndarray] = []

    for seed in seeds:
        for fold_idx, tr_idx, va_idx in _split_iter(cfg, train_df, y_proxy):
            done_marker = folds_dir / f"done_seed{seed}_fold{fold_idx}.json"
            if done_marker.exists():
                payload = json.loads(done_marker.read_text())
                per_fold_scores.append(float(payload["score"]))
                _append_progress(
                    progress_log,
                    {"stage": "stage2", "event": "fold_skipped", "run_id": cfg["run_id"],
                     "seed": seed, "fold": fold_idx},
                )
                continue

            _append_progress(
                progress_log,
                {"stage": "stage2", "event": "fold_started", "run_id": cfg["run_id"],
                 "seed": seed, "fold": fold_idx},
            )
            t0 = time.time()

            paths = convert_annotations_for_framework(cfg, train_df, (tr_idx, va_idx))
            score, test_pred = train_one_fold(cfg, paths)
            per_fold_scores.append(score)
            test_preds_all.append(test_pred)
            np.save(folds_dir / f"test_seed{seed}_fold{fold_idx}.npy", test_pred)
            done_marker.write_text(json.dumps({"score": score}))

            _append_progress(
                progress_log,
                {"stage": "stage2", "event": "fold_done", "run_id": cfg["run_id"],
                 "seed": seed, "fold": fold_idx, "score": score,
                 "wallclock_s": time.time() - t0},
            )

    cv_mean = float(np.mean(per_fold_scores)) if per_fold_scores else 0.0
    cv_std = float(np.std(per_fold_scores)) if per_fold_scores else 0.0
    metric_name = (cfg.get("metric") or {}).get("name", "map")
    (run_dir / "cv_score.json").write_text(
        json.dumps(
            {
                "metric": metric_name,
                "per_fold": per_fold_scores,
                "mean": cv_mean,
                "std": cv_std,
                "seeds": seeds,
                "n_folds": n_folds,
                "framework": fw,
            },
            indent=2,
        )
    )

    # Aggregate test predictions (framework-specific; agent's encode_submission handles it).
    if test_preds_all:
        try:
            sub = encode_submission(cfg, test_preds_all)
            sub.to_csv(run_dir / "test_preds.csv", index=False)
        except NotImplementedError as e:
            sys.stderr.write(f"encode_submission not implemented: {e}\n")
            return 5

    print(f"CV {metric_name} = {cv_mean:.6f} (std {cv_std:.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
