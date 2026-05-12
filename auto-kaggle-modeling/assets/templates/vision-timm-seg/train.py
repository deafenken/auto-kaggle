"""Segmentation training template — SKELETON, agent customizes per competition.

Same shape as vision-timm/train.py: the outer training loop is functional, and
the dataset / model / metric functions raise NotImplementedError. The agent
edits the run-dir copy.

Designed around segmentation_models_pytorch (smp) by default — wraps any timm
encoder in a U-Net/FPN/DeepLabV3+ decoder. Override `framework: custom` and
write your own `build_model` if smp does not fit the comp.

Invocation: same as other templates.
    python train.py --config config.yaml --run-dir <abs> --progress-log <abs>
"""

from __future__ import annotations

import argparse
import hashlib
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


def _config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Agent customizes these per competition.
# ---------------------------------------------------------------------------


def build_model(cfg: dict[str, Any]):
    """Return a torch.nn.Module mapping image → (B, classes, H, W) logits."""
    fw = cfg.get("framework", "smp")
    if fw == "smp":
        # Default smp wrapper. Agent overrides `classes` and possibly `architecture`.
        import segmentation_models_pytorch as smp  # noqa: PLC0415

        arch_cls = getattr(smp, cfg["architecture"])
        return arch_cls(
            encoder_name=cfg["encoder_name"],
            encoder_weights=cfg.get("encoder_weights", "imagenet"),
            in_channels=cfg.get("in_channels", 3),
            classes=int(cfg["classes"]),
        )
    raise NotImplementedError(
        f"framework '{fw}' is not in skeleton — agent must implement build_model()"
    )


def build_datasets(cfg: dict[str, Any], train_df: pd.DataFrame, fold_indices):
    """Return (train_dataset, val_dataset). Items yield dict with 'image' (FloatTensor)
    and 'mask' (FloatTensor or LongTensor, shape depends on n_classes)."""
    raise NotImplementedError(
        "agent must implement build_datasets() for the comp's image/mask paths "
        "and augmentation pipeline (albumentations with mask=True)."
    )


def build_test_dataset(cfg: dict[str, Any], test_df: pd.DataFrame):
    raise NotImplementedError("agent must implement build_test_dataset()")


def compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Segmentation metrics: IoU / Dice / per-class variants."""
    m = metric_name.lower()
    if m in ("iou", "jaccard"):
        return _iou_score(y_true, y_pred)
    if m == "dice":
        return _dice_score(y_true, y_pred)
    raise NotImplementedError(
        f"metric '{metric_name}' not in skeleton — agent must implement compute_metric()"
    )


def encode_submission(cfg: dict[str, Any], test_ids, masks_logits) -> pd.DataFrame:
    """Turn predicted logits into the comp's submission format (RLE or other)."""
    raise NotImplementedError(
        "agent must implement encode_submission() — RLE vs PNG paths vs other "
        "depends on the competition's sample_submission.csv columns."
    )


def _iou_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    p = (y_pred > threshold).astype(np.uint8)
    t = (y_true > threshold).astype(np.uint8)
    inter = float((p & t).sum())
    union = float((p | t).sum()) + 1e-9
    return inter / union


def _dice_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    p = (y_pred > threshold).astype(np.uint8)
    t = (y_true > threshold).astype(np.uint8)
    inter = float((p & t).sum())
    return (2 * inter + 1e-9) / (float(p.sum() + t.sum()) + 1e-9)


def _build_loss(cfg: dict[str, Any]):
    import torch.nn as nn  # noqa: PLC0415

    spec = cfg["loss"] or {}
    kind = spec.get("type", "bce_dice")
    try:
        import segmentation_models_pytorch as smp  # noqa: PLC0415
    except ImportError:
        smp = None  # type: ignore

    if kind == "dice" and smp is not None:
        return smp.losses.DiceLoss(mode="binary" if int(cfg["classes"]) == 1 else "multiclass")
    if kind == "focal" and smp is not None:
        return smp.losses.FocalLoss(mode="binary" if int(cfg["classes"]) == 1 else "multiclass")
    if kind == "lovasz" and smp is not None:
        return smp.losses.LovaszLoss(mode="binary" if int(cfg["classes"]) == 1 else "multiclass")
    if kind == "bce_dice" and smp is not None:
        bce = nn.BCEWithLogitsLoss()
        dice = smp.losses.DiceLoss(mode="binary" if int(cfg["classes"]) == 1 else "multiclass")
        w_bce = float(spec.get("bce_weight", 0.5))
        w_dice = float(spec.get("dice_weight", 0.5))

        def loss_fn(logits, target):
            return w_bce * bce(logits, target.float()) + w_dice * dice(logits, target)

        return loss_fn
    # Fallback: BCE only
    return nn.BCEWithLogitsLoss()


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

    seeds = cfg.get("seeds", [42])
    n_folds = int(cfg["cv_split"]["n_folds"])
    folds_dir = run_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(cfg["data_paths"]["train"])
    test_df = pd.read_csv(cfg["data_paths"]["test"])
    sample_sub = pd.read_csv(cfg["data_paths"]["sample_submission"])

    fc = cfg.get("feature_config") or {}
    target_col = fc.get("target_col", "mask_path")
    id_col = fc.get("id_col", "image_id")
    y_proxy = train_df.index.to_numpy()  # for shape; CV by index/group only

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _cv_common import _split_iter  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_name = (cfg.get("metric") or {}).get("name", "dice")
    per_fold_scores: list[float] = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for fold_idx, tr_idx, va_idx in _split_iter(cfg, train_df, y_proxy):
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
            tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=cfg.get("num_workers", 4), pin_memory=True)
            vl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg.get("num_workers", 4), pin_memory=True)
            tel = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg.get("num_workers", 4), pin_memory=True)

            model = build_model(cfg).to(device)
            loss_fn = _build_loss(cfg)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0)
            )
            total_steps = cfg["epochs"] * len(tl)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
            scaler = torch.cuda.amp.GradScaler() if cfg.get("mixed_precision") else None

            for epoch in range(cfg["epochs"]):
                model.train()
                for batch in tl:
                    x = batch["image"].to(device, non_blocking=True)
                    y = batch["mask"].to(device, non_blocking=True)
                    optimizer.zero_grad()
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            logits = model(x)
                            loss = loss_fn(logits, y)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = model(x)
                        loss = loss_fn(logits, y)
                        loss.backward()
                        optimizer.step()
                    scheduler.step()
                _append_progress(
                    progress_log,
                    {"stage": "stage2", "event": "epoch_done", "run_id": cfg["run_id"],
                     "seed": seed, "fold": fold_idx, "epoch": epoch},
                )

            # Inference — store probabilities; encoding to submission format happens after.
            val_probs = _infer(model, vl, device, cfg)
            test_probs = _infer(model, tel, device, cfg)
            np.save(slice_path, val_probs.astype(np.float16))   # save space — masks are large
            np.save(test_slice_path, test_probs.astype(np.float16))
            cfg_hash_path.write_text(cfg_hash)

            # CV metric: agent's compute_metric on probabilities vs ground truth masks.
            # The val dataset must expose true masks via val_ds.get_true_masks() in the agent's impl.
            true_masks = getattr(val_ds, "true_masks", None)
            if true_masks is None:
                # Skeleton fallback — needs implementation.
                fold_score = float("nan")
            else:
                fold_score = compute_metric(metric_name, np.asarray(true_masks), val_probs)
            per_fold_scores.append(fold_score)
            _append_progress(
                progress_log,
                {"stage": "stage2", "event": "fold_done", "run_id": cfg["run_id"],
                 "seed": seed, "fold": fold_idx, "score": fold_score,
                 "wallclock_s": time.time() - t0},
            )

    cv_mean = float(np.nanmean(per_fold_scores)) if per_fold_scores else 0.0
    cv_std = float(np.nanstd(per_fold_scores)) if per_fold_scores else 0.0
    (run_dir / "cv_score.json").write_text(
        json.dumps(
            {
                "metric": metric_name,
                "per_fold": per_fold_scores,
                "mean": cv_mean,
                "std": cv_std,
                "seeds": seeds,
                "n_folds": n_folds,
            },
            indent=2,
        )
    )

    # Submission encoding — agent fills this in.
    print(
        f"CV {metric_name} = {cv_mean:.6f} (std {cv_std:.6f}); "
        f"call encode_submission() to produce test_preds.csv"
    )
    return 0


def _infer(model, loader, device, cfg):
    import torch  # noqa: PLC0415

    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits) if int(cfg["classes"]) == 1 else logits.softmax(dim=1)
            outs.append(probs.cpu().numpy())
    return np.concatenate(outs, axis=0)


if __name__ == "__main__":
    sys.exit(main())
