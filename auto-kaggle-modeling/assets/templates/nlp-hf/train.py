"""NLP training template — SKELETON, agent must complete per competition.

Same shape as vision-timm/train.py: outer loop is functional, the model + dataset
+ metric functions raise NotImplementedError and must be implemented in the run-
dir copy.

HuggingFace Transformers is the default toolkit. The skeleton uses Trainer-style
loops by hand (no `Trainer` class) because we need fold-by-fold crash safety,
which `Trainer` doesn't give us out of the box.

Invocation:
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_progress(progress_log: Path, event: dict[str, Any]) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts_utc": _now_iso(), **event}
    with progress_log.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def build_model_and_tokenizer(cfg: dict[str, Any]):
    """Return (model, tokenizer). Default: AutoModelForSequenceClassification.

    Override for token classification, regression head, seq2seq, etc.
    """
    raise NotImplementedError(
        "agent must implement build_model_and_tokenizer() in the run-dir copy — "
        "the right HF class depends on task: AutoModelForSequenceClassification, "
        "AutoModelForTokenClassification, AutoModelForSeq2SeqLM, etc."
    )


def tokenize_batch(batch: dict[str, Any], tokenizer, cfg: dict[str, Any]):
    """Return a dict with input_ids, attention_mask, labels.

    Default: single-text tokenization, target column → labels.
    """
    raise NotImplementedError(
        "agent must implement tokenize_batch() for the comp's text schema "
        "(single text vs pair text, label encoding for token classification, etc.)."
    )


def compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import (  # noqa: PLC0415
        f1_score,
        mean_squared_error,
        roc_auc_score,
    )

    m = metric_name.lower()
    if m == "auc":
        return float(roc_auc_score(y_true, y_pred))
    if m == "f1":
        return float(f1_score(y_true, (y_pred > 0.5).astype(int)))
    if m == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    raise NotImplementedError(
        f"metric '{metric_name}' not in default NLP template — implement compute_metric()."
    )


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
    from torch.utils.data import DataLoader, Dataset  # noqa: PLC0415
    from transformers import get_linear_schedule_with_warmup  # noqa: PLC0415

    metric_name = (cfg.get("metric") or {}).get("name", "auc")
    seeds = cfg.get("seeds", [42])
    n_folds = int(cfg["cv_split"]["n_folds"])
    folds_dir = run_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(cfg["data_paths"]["train"])
    test_df = pd.read_csv(cfg["data_paths"]["test"])
    sample_sub = pd.read_csv(cfg["data_paths"]["sample_submission"])

    fc = cfg.get("feature_config") or {}
    target_col = fc.get("target_col", "label")
    id_col = fc.get("id_col", "id")
    y_train = train_df[target_col].to_numpy()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _cv_common import _split_iter  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_fold_scores: list[float] = []

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

            model, tokenizer = build_model_and_tokenizer(cfg)
            model = model.to(device)

            train_subset = train_df.iloc[tr_idx].reset_index(drop=True)
            val_subset = train_df.iloc[va_idx].reset_index(drop=True)

            train_loader = _make_loader(train_subset, tokenizer, cfg, shuffle=True)
            val_loader = _make_loader(val_subset, tokenizer, cfg, shuffle=False)
            test_loader = _make_loader(test_df, tokenizer, cfg, shuffle=False, has_target=False)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0)
            )
            total_steps = cfg["epochs"] * len(train_loader)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_steps * cfg.get("warmup_ratio", 0.0)),
                num_training_steps=total_steps,
            )
            scaler = torch.cuda.amp.GradScaler() if cfg.get("mixed_precision") else None

            for epoch in range(cfg["epochs"]):
                model.train()
                for step, batch in enumerate(train_loader):
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    optimizer.zero_grad()
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            out = model(**batch)
                            loss = out.loss
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        out = model(**batch)
                        loss = out.loss
                        loss.backward()
                        optimizer.step()
                    scheduler.step()
                _append_progress(
                    progress_log,
                    {"stage": "stage2", "event": "epoch_done", "run_id": cfg["run_id"],
                     "seed": seed, "fold": fold_idx, "epoch": epoch},
                )

            val_preds = _infer(model, val_loader, device)
            test_preds = _infer(model, test_loader, device)
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

    cv_mean = float(np.mean(per_fold_scores)) if per_fold_scores else 0.0
    cv_std = float(np.std(per_fold_scores)) if per_fold_scores else 0.0
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
    print(f"CV {metric_name} = {cv_mean:.6f} (std {cv_std:.6f})")
    return 0


def _make_loader(df: pd.DataFrame, tokenizer, cfg: dict[str, Any], shuffle: bool, has_target: bool = True):
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, Dataset  # noqa: PLC0415

    fc = cfg["feature_config"]
    text_col = fc["text_col"]
    target_col = fc.get("target_col")
    max_len = int(cfg["max_seq_length"])

    class _Ds(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, idx):
            row = df.iloc[idx]
            enc = tokenizer(
                str(row[text_col]),
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            if has_target and target_col in row:
                item["labels"] = torch.tensor(row[target_col])
            return item

    return DataLoader(
        _Ds(),
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )


def _infer(model, loader, device):
    import torch  # noqa: PLC0415

    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "labels"}
            out = model(**batch)
            logits = out.logits
            if logits.ndim == 2 and logits.shape[-1] == 1:
                outs.append(logits.squeeze(-1).cpu().numpy())
            else:
                outs.append(logits.softmax(dim=-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


if __name__ == "__main__":
    sys.exit(main())
