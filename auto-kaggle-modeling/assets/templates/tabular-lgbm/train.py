"""Tabular LightGBM training template.

Invoked by modeling.py as a subprocess from inside the run's directory:

    python train.py --config config.yaml --run-dir <abs path> --progress-log <abs path>

Responsibilities (contract from auto-kaggle-modeling/SKILL.md Step 5):
  1. Read CV scheme + data paths from config.yaml.
  2. Train one OOF per fold (and per seed if seeds list has > 1 entry).
  3. After each fold completes, atomically save the fold's OOF slice + test preds.
  4. Append `fold_started` and `fold_done` events to progress.jsonl with per-fold CV.
  5. On all-folds-done: concatenate OOFs, compute aggregate CV, write the
     run-level outputs the contract specifies (oof.npy, test_preds.csv, cv_score.json).

Resumes: if a fold's slice file already exists with a matching config hash,
that fold is skipped (idempotency for fold-level checkpointing).
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
    norm = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(norm.encode()).hexdigest()[:10]


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp.npy")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.save(tmp, arr)
    os.replace(tmp, path)


def _atomic_save_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _split_iter(cfg: dict[str, Any], df: pd.DataFrame, y: np.ndarray):
    """Generate (fold_idx, train_idx, val_idx) tuples per cv_split.yaml."""
    from sklearn.model_selection import (  # noqa: PLC0415
        GroupKFold,
        KFold,
        StratifiedKFold,
    )

    cv = cfg["cv_split"]
    scheme = cv["scheme"]
    n_folds = int(cv["n_folds"])
    seed = cv.get("seed")
    if scheme == "StratifiedKFold":
        sk = StratifiedKFold(
            n_splits=n_folds, shuffle=bool(cv.get("shuffle", True)), random_state=seed
        )
        for i, (tr, va) in enumerate(sk.split(df, y)):
            yield i, tr, va
    elif scheme == "KFold":
        kf = KFold(
            n_splits=n_folds, shuffle=bool(cv.get("shuffle", True)), random_state=seed
        )
        for i, (tr, va) in enumerate(kf.split(df)):
            yield i, tr, va
    elif scheme == "GroupKFold":
        gk = GroupKFold(n_splits=n_folds)
        for i, (tr, va) in enumerate(gk.split(df, y, groups=df[cv["group_col"]])):
            yield i, tr, va
    else:
        raise ValueError(
            f"unsupported CV scheme '{scheme}' for this template — write a custom "
            "train.py or extend train.py."
        )


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        roc_auc_score,
    )

    m = metric_name.lower()
    if m == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if m == "rmsle":
        # Predictions must be non-negative for log1p; clip to be safe.
        p = np.clip(y_pred, 0, None)
        return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(p))))
    if m == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if m == "auc":
        return float(roc_auc_score(y_true, y_pred))
    if m == "log_loss":
        return float(log_loss(y_true, y_pred))
    if m == "accuracy":
        return float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))
    if m == "f1":
        return float(f1_score(y_true, (y_pred > 0.5).astype(int)))
    # Fallback — RMSE
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _apply_target_transform(y: np.ndarray, transform: str | None) -> tuple[np.ndarray, callable]:
    if transform is None:
        return y, lambda x: x
    if transform == "log1p":
        return np.log1p(y), np.expm1
    if transform == "boxcox":
        from scipy.stats import boxcox  # type: ignore  # noqa: PLC0415

        y_t, lam = boxcox(y + 1e-9)
        return y_t, lambda x: np.power(np.maximum(x * lam + 1, 1e-12), 1.0 / lam) - 1e-9
    raise ValueError(f"unknown target_transform: {transform}")


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

    fc = cfg.get("feature_config") or {}
    target_col = fc.get("target_col", "target")
    id_col = fc.get("id_col", "id")
    drop_cols = set(fc.get("drop_cols", []) or [])

    train = pd.read_csv(cfg["data_paths"]["train"])
    test = pd.read_csv(cfg["data_paths"]["test"])
    sample_sub = pd.read_csv(cfg["data_paths"]["sample_submission"])

    y_raw = train[target_col].to_numpy()
    y, inverse = _apply_target_transform(y_raw, fc.get("target_transform"))

    feature_cols = [
        c for c in train.columns if c not in drop_cols and c != target_col and c != id_col
    ]
    # Categorical detection: object dtype → category
    cat_cols = [c for c in feature_cols if train[c].dtype == object]
    for c in cat_cols:
        train[c] = train[c].astype("category")
        if c in test.columns:
            test[c] = test[c].astype("category")
            # Align categories across train/test so LightGBM sees same encoding.
            test[c] = test[c].cat.set_categories(train[c].cat.categories)

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    metric_name = (cfg.get("metric") or {}).get("name", "rmse")
    metric_dir = (cfg.get("metric") or {}).get("direction", "minimize")
    task_type = cfg.get("task_type", "tabular-regression")

    # LightGBM params
    lgb_params = {
        "objective": _lgbm_objective(task_type, metric_name),
        "metric": _lgbm_metric(metric_name),
        "num_leaves": cfg["num_leaves"],
        "learning_rate": cfg["learning_rate"],
        "feature_fraction": cfg["feature_fraction"],
        "bagging_fraction": cfg["bagging_fraction"],
        "bagging_freq": cfg["bagging_freq"],
        "min_data_in_leaf": cfg["min_data_in_leaf"],
        "verbose": -1,
        "n_jobs": -1,
    }
    if cfg.get("device") == "gpu":
        lgb_params["device"] = "gpu"

    seeds = cfg.get("seeds", [42])
    n_folds = int(cfg["cv_split"]["n_folds"])
    folds_dir = run_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    # OOF and test prediction accumulators (averaged over seeds at the end).
    n_train = len(train)
    n_test = len(test)
    if task_type == "tabular-multiclass":
        n_classes = len(np.unique(y_raw))
        oof_acc = np.zeros((n_train, n_classes), dtype=np.float64)
        test_acc = np.zeros((n_test, n_classes), dtype=np.float64)
    else:
        n_classes = 1
        oof_acc = np.zeros(n_train, dtype=np.float64)
        test_acc = np.zeros(n_test, dtype=np.float64)

    import lightgbm as lgb  # noqa: PLC0415  (defer so dry-runs don't require it)

    per_fold_scores: list[float] = []

    for seed_idx, seed in enumerate(seeds):
        for fold_idx, tr_idx, va_idx in _split_iter(cfg, train, y_raw):
            slice_path = folds_dir / f"oof_seed{seed}_fold{fold_idx}.npy"
            test_slice_path = folds_dir / f"test_seed{seed}_fold{fold_idx}.npy"
            cfg_hash_path = folds_dir / f"cfg_seed{seed}_fold{fold_idx}.hash"
            # Resume: skip if already done with matching cfg hash.
            if (
                slice_path.exists()
                and test_slice_path.exists()
                and cfg_hash_path.exists()
                and cfg_hash_path.read_text().strip() == cfg_hash
            ):
                _append_progress(
                    progress_log,
                    {
                        "stage": "stage2",
                        "event": "fold_skipped",
                        "run_id": cfg["run_id"],
                        "seed": seed,
                        "fold": fold_idx,
                    },
                )
                continue

            _append_progress(
                progress_log,
                {
                    "stage": "stage2",
                    "event": "fold_started",
                    "run_id": cfg["run_id"],
                    "seed": seed,
                    "fold": fold_idx,
                },
            )
            t0 = time.time()

            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            params = {**lgb_params, "seed": seed}
            train_set = lgb.Dataset(
                X_tr,
                label=y_tr,
                categorical_feature=cat_cols if cat_cols else "auto",
            )
            val_set = lgb.Dataset(
                X_va,
                label=y_va,
                reference=train_set,
                categorical_feature=cat_cols if cat_cols else "auto",
            )
            model = lgb.train(
                params,
                train_set,
                num_boost_round=cfg["n_estimators"],
                valid_sets=[train_set, val_set],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(cfg["early_stopping_rounds"]),
                    lgb.log_evaluation(period=0),
                ],
            )
            pred_va = model.predict(X_va, num_iteration=model.best_iteration)
            pred_test = model.predict(X_test, num_iteration=model.best_iteration)
            # Apply inverse transform if any.
            pred_va_orig = inverse(pred_va) if fc.get("target_transform") else pred_va
            pred_test_orig = inverse(pred_test) if fc.get("target_transform") else pred_test

            # Atomic per-fold save (so a crash mid-next-fold is recoverable).
            _atomic_save_npy(slice_path, np.asarray(pred_va_orig))
            _atomic_save_npy(test_slice_path, np.asarray(pred_test_orig))
            cfg_hash_path.write_text(cfg_hash)

            fold_score = _compute_metric(metric_name, y_raw[va_idx], pred_va_orig)
            per_fold_scores.append(fold_score)
            took = time.time() - t0
            _append_progress(
                progress_log,
                {
                    "stage": "stage2",
                    "event": "fold_done",
                    "run_id": cfg["run_id"],
                    "seed": seed,
                    "fold": fold_idx,
                    "score": fold_score,
                    "best_iter": model.best_iteration,
                    "wallclock_s": took,
                },
            )

    # Aggregate: average OOFs / test preds across seeds and folds.
    for seed in seeds:
        for fold_idx in range(n_folds):
            slice_path = folds_dir / f"oof_seed{seed}_fold{fold_idx}.npy"
            test_slice_path = folds_dir / f"test_seed{seed}_fold{fold_idx}.npy"
            if not slice_path.exists() or not test_slice_path.exists():
                sys.stderr.write(
                    f"BUDGET_WARN or crash mid-fold left fold {fold_idx} of seed {seed} incomplete\n"
                )
                return 5  # partial — caller will resume

    # Recompute OOF / test from disk (this is the source of truth on resume).
    oof_full = np.zeros_like(oof_acc) if n_classes > 1 else np.zeros(n_train, dtype=np.float64)
    test_full = np.zeros_like(test_acc) if n_classes > 1 else np.zeros(n_test, dtype=np.float64)
    seeds_count = len(seeds)
    for seed in seeds:
        for fold_idx, tr_idx, va_idx in _split_iter(cfg, train, y_raw):
            sl = np.load(folds_dir / f"oof_seed{seed}_fold{fold_idx}.npy")
            ts = np.load(folds_dir / f"test_seed{seed}_fold{fold_idx}.npy")
            oof_full[va_idx] += sl / seeds_count
            test_full += ts / (n_folds * seeds_count)

    _atomic_save_npy(run_dir / "oof.npy", oof_full)

    # Write test predictions in submission format.
    sub = sample_sub.copy()
    sub_cols = [c for c in sub.columns if c.lower() != id_col.lower()]
    if len(sub_cols) == 1:
        sub[sub_cols[0]] = test_full
    else:
        # Multi-column submission — assume model emits one column per sub col.
        for i, c in enumerate(sub_cols):
            sub[c] = test_full[:, i] if test_full.ndim > 1 else test_full
    _atomic_save_csv(run_dir / "test_preds.csv", sub)

    # CV summary.
    cv_mean = float(np.mean(per_fold_scores)) if per_fold_scores else 0.0
    cv_std = float(np.std(per_fold_scores)) if per_fold_scores else 0.0
    cv_summary = {
        "metric": metric_name,
        "direction": metric_dir,
        "per_fold": per_fold_scores,
        "mean": cv_mean,
        "std": cv_std,
        "n_train": n_train,
        "n_test": n_test,
        "seeds": seeds,
        "n_folds": n_folds,
    }
    (run_dir / "cv_score.json").write_text(json.dumps(cv_summary, indent=2))
    print(f"CV {metric_name} = {cv_mean:.6f} (std {cv_std:.6f})")
    return 0


def _lgbm_objective(task_type: str, metric_name: str) -> str:
    if task_type == "tabular-binary":
        return "binary"
    if task_type == "tabular-multiclass":
        return "multiclass"
    if task_type == "tabular-ranking":
        return "lambdarank"
    if metric_name.lower() == "mae":
        return "regression_l1"
    return "regression"


def _lgbm_metric(name: str) -> str:
    m = name.lower()
    if m in ("auc", "rmse", "mae", "binary_logloss", "multi_logloss"):
        return m
    if m == "log_loss":
        return "binary_logloss"
    if m == "rmsle":
        return "rmse"  # we apply log1p in metric calc, LGBM optimizes plain rmse
    return "rmse"


if __name__ == "__main__":
    sys.exit(main())
