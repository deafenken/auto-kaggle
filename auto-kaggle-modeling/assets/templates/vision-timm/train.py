"""Vision training template — SKELETON, agent must complete per competition.

This file is intentionally a skeleton. timm + albumentations + torch can vary
substantially across competitions (image loading paths, mask formats, target
column semantics, batch composition for segmentation). Rather than hardcoding
one flavor, this file:

  1. Sets up the outer training loop, optimizer, scheduler, mixed precision.
  2. Leaves `build_model()`, `load_train_item()`, `load_test_item()`,
     `compute_metric()` as functions the agent fills in for the specific comp.
  3. Honors the same contract as tabular-lgbm: fold-by-fold checkpoints,
     atomic saves, progress.jsonl events on each fold, cv_score.json at end.

The agent EDITS THE COPY in runs/<comp_slug>/stage2_modeling/runs/<run_id>/train.py,
never this source. The skeleton is a starting point; the run-specific version
is where the actual model architecture and metric live.

Invocation (same as tabular-lgbm):
    python train.py --config config.yaml --run-dir <abs> --progress-log <abs>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Heavy imports deferred so dry-runs / lint don't fail.


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_progress(progress_log: Path, event: dict[str, Any]) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts_utc": _now_iso(), **event}
    with progress_log.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Functions the agent MUST customize per competition. The skeleton versions
# raise NotImplementedError so a forgotten customization is loud, not silent.
# ---------------------------------------------------------------------------


def build_model(cfg: dict[str, Any]):
    """Return a torch.nn.Module ready for training.

    Default (single-label classification): timm.create_model with
    num_classes inferred from the train DataFrame target column.
    """
    raise NotImplementedError(
        "agent must implement build_model() in the run-dir copy of train.py — "
        "the skeleton needs to know the number of classes / output shape, and "
        "any custom head (segmentation, multi-label, etc.)."
    )


def build_datasets(cfg: dict[str, Any], train_df: pd.DataFrame, fold_indices):
    """Return (train_dataset, val_dataset). Caller wraps in DataLoaders."""
    raise NotImplementedError(
        "agent must implement build_datasets() — image loading + augmentation pipeline."
    )


def build_test_dataset(cfg: dict[str, Any], test_df: pd.DataFrame):
    raise NotImplementedError("agent must implement build_test_dataset()")


def compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition-specific metric. Defaults to accuracy / AUC if standard."""
    from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: PLC0415

    m = metric_name.lower()
    if m == "accuracy":
        return float(accuracy_score(y_true, np.argmax(y_pred, axis=-1)))
    if m == "auc":
        return float(roc_auc_score(y_true, y_pred))
    raise NotImplementedError(
        f"metric '{metric_name}' is not in the default vision template — "
        "implement compute_metric() for this competition."
    )


# ---------------------------------------------------------------------------
# Outer training loop — usually no customization needed.
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
    cfg_hash = _config_hash(cfg)

    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    metric_name = (cfg.get("metric") or {}).get("name", "accuracy")
    seeds = cfg.get("seeds", [42])
    n_folds = int(cfg["cv_split"]["n_folds"])
    folds_dir = run_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(cfg["data_paths"]["train"])
    test_df = pd.read_csv(cfg["data_paths"]["test"])
    sample_sub = pd.read_csv(cfg["data_paths"]["sample_submission"])

    fc = cfg.get("feature_config") or {}
    target_col = fc.get("target_col", "label")
    id_col = fc.get("id_col", "image_id")
    y_train = train_df[target_col].to_numpy()

    # CV split — same scheme dispatch as tabular template
    sys.path.insert(0, str(Path(__file__).parent.parent / "tabular-lgbm"))
    from train import _split_iter  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_fold_scores: list[float] = []
    n_train = len(train_df)
    n_test = len(test_df)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for fold_idx, tr_idx, va_idx in _split_iter(cfg, train_df, y_train):
            slice_path = folds_dir / f"oof_seed{seed}_fold{fold_idx}.npy"
            test_slice_path = folds_dir / f"test_seed{seed}_fold{fold_idx}.npy"
            cfg_hash_path = folds_dir / f"cfg_seed{seed}_fold{fold_idx}.hash"
            if (
                slice_path.exists()
                and test_slice_path.exists()
                and cfg_hash_path.exists()
                and cfg_hash_path.read_text().strip() == cfg_hash
            ):
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

            train_ds, val_ds = build_datasets(cfg, train_df, (tr_idx, va_idx))
            test_ds = build_test_dataset(cfg, test_df)
            train_loader = DataLoader(
                train_ds, batch_size=cfg["batch_size"], shuffle=True,
                num_workers=cfg.get("num_workers", 4), pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds, batch_size=cfg["batch_size"], shuffle=False,
                num_workers=cfg.get("num_workers", 4), pin_memory=True,
            )
            test_loader = DataLoader(
                test_ds, batch_size=cfg["batch_size"], shuffle=False,
                num_workers=cfg.get("num_workers", 4), pin_memory=True,
            )

            model = build_model(cfg).to(device)
            optimizer = _build_optimizer(model, cfg)
            scheduler = _build_scheduler(optimizer, cfg, len(train_loader))
            scaler = torch.cuda.amp.GradScaler() if cfg.get("mixed_precision") else None

            for epoch in range(cfg["epochs"]):
                _train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg)
                _append_progress(
                    progress_log,
                    {"stage": "stage2", "event": "epoch_done", "run_id": cfg["run_id"],
                     "seed": seed, "fold": fold_idx, "epoch": epoch},
                )

            val_preds = _infer(model, val_loader, device, cfg)
            test_preds = _infer(model, test_loader, device, cfg)
            np.save(slice_path, val_preds)
            np.save(test_slice_path, test_preds)
            cfg_hash_path.write_text(cfg_hash)

            fold_score = compute_metric(metric_name, y_train[va_idx], val_preds)
            per_fold_scores.append(fold_score)
            _append_progress(
                progress_log,
                {"stage": "stage2", "event": "fold_done", "run_id": cfg["run_id"],
                 "seed": seed, "fold": fold_idx, "score": fold_score,
                 "wallclock_s": time.time() - t0},
            )

    # Aggregate (same pattern as tabular-lgbm template)
    # ... agent fills in for the specific output shape

    cv_mean = float(np.mean(per_fold_scores)) if per_fold_scores else 0.0
    cv_std = float(np.std(per_fold_scores)) if per_fold_scores else 0.0
    (run_dir / "cv_score.json").write_text(
        json.dumps(
            {
                "metric": metric_name,
                "per_fold": per_fold_scores,
                "mean": cv_mean,
                "std": cv_std,
                "n_train": n_train,
                "n_test": n_test,
                "seeds": seeds,
                "n_folds": n_folds,
            },
            indent=2,
        )
    )
    print(f"CV {metric_name} = {cv_mean:.6f} (std {cv_std:.6f})")
    return 0


def _build_optimizer(model, cfg):
    import torch  # noqa: PLC0415

    name = cfg.get("optimizer", "adamw").lower()
    params = {"lr": cfg["lr"], "weight_decay": cfg.get("weight_decay", 0)}
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **params)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **params)
    raise ValueError(f"unknown optimizer: {name}")


def _build_scheduler(optimizer, cfg, steps_per_epoch):
    import torch  # noqa: PLC0415

    name = cfg.get("scheduler", "cosine").lower()
    total_steps = cfg["epochs"] * steps_per_epoch
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if name == "none":
        return None
    raise ValueError(f"unknown scheduler: {name}")


def _train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg):
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    model.train()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = _default_loss(logits, y, cfg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = _default_loss(logits, y, cfg)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()


def _default_loss(logits, y, cfg):
    import torch.nn.functional as F  # noqa: PLC0415

    task = cfg.get("task_type", "image-classification")
    if task == "image-classification":
        return F.cross_entropy(logits, y)
    raise NotImplementedError(f"default loss for task '{task}' not in skeleton")


@torch.no_grad if False else (lambda f: f)  # placeholder for type checkers
def _infer(model, loader, device, cfg):
    import torch  # noqa: PLC0415

    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            outs.append(model(x).softmax(dim=-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


if __name__ == "__main__":
    sys.exit(main())
