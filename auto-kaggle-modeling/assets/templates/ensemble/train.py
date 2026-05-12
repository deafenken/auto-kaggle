"""Ensemble engine — blend or stack the OOFs / test preds of prior runs.

This template does not train a model on raw data. Its "training" is fitting a
meta-learner on the OOF matrix (mode=stack) or computing a weighted aggregation
(mode=blend). It produces the same artifacts the contract requires:
oof.npy, test_preds.csv, cv_score.json, attribution.md (auto-extended).

Invocation:
    python train.py --config config.yaml --run-dir <abs> --progress-log <abs>

Required config fields:
  - mode: 'blend' or 'stack'
  - source_runs: list of run_ids whose oof.npy + test_preds.csv to use
  - blend.* or stack.* depending on mode
  - cv_split: same as other templates (used for stack meta-learner CV)
  - metric.name + metric.direction
  - data_paths.train (for the target column) + data_paths.sample_submission

Source runs are read from:
    runs/<comp_slug>/stage2_modeling/runs/<source_run_id>/oof.npy
    runs/<comp_slug>/stage2_modeling/runs/<source_run_id>/test_preds.csv

If any source run is missing oof.npy or test_preds.csv, the script refuses
to run (rather than silently dropping it).

The ensemble run's `attribution.md` should be pre-written by modeling.py with
the source runs listed. This script appends an "Ensemble construction" section
documenting weights / meta-learner choices.
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


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp.npy")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.save(tmp, arr)
    tmp.replace(path)


def _atomic_save_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _resolve_source_run_dir(run_dir: Path, source_run_id: str) -> Path:
    """Sibling directory in stage2_modeling/runs/."""
    return run_dir.parent / source_run_id


def _load_source(run_dir: Path, source_run_id: str) -> tuple[np.ndarray, pd.DataFrame, str, float]:
    src = _resolve_source_run_dir(run_dir, source_run_id)
    oof_path = src / "oof.npy"
    preds_path = src / "test_preds.csv"
    cv_path = src / "cv_score.json"
    for p in (oof_path, preds_path, cv_path):
        if not p.exists():
            raise FileNotFoundError(
                f"source run {source_run_id} is missing {p.name}; ensemble refuses to proceed"
            )
    oof = np.load(oof_path)
    preds = pd.read_csv(preds_path)
    cv = json.loads(cv_path.read_text())
    return oof, preds, cv.get("metric", "metric"), float(cv["mean"])


def _rank_within_each(arr: np.ndarray) -> np.ndarray:
    """Rank values column-wise, normalized to [0, 1]. arr is (n,) or (n, k)."""
    if arr.ndim == 1:
        order = arr.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(arr))
        return ranks / max(len(arr) - 1, 1)
    out = np.empty_like(arr, dtype=float)
    for k in range(arr.shape[1]):
        order = arr[:, k].argsort()
        r = np.empty(arr.shape[0], dtype=float)
        r[order] = np.arange(arr.shape[0])
        out[:, k] = r / max(arr.shape[0] - 1, 1)
    return out


def _blend(oofs: list[np.ndarray], tests: list[np.ndarray], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict]:
    bcfg = cfg["blend"] or {}
    agg = bcfg.get("aggregation", "arithmetic")
    weights = bcfg.get("weights")
    n = len(oofs)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(f"weights length {len(weights)} != source_runs length {n}")
        s = float(sum(weights))
        if s <= 0:
            raise ValueError("weights must sum > 0")
        weights = [w / s for w in weights]

    if agg == "arithmetic":
        oof_blend = sum(w * o for w, o in zip(weights, oofs))
        test_blend = sum(w * t for w, t in zip(weights, tests))
    elif agg == "geometric":
        eps = float(cfg.get("geometric_eps", 1e-9))
        log_oof = sum(w * np.log(np.maximum(o, eps)) for w, o in zip(weights, oofs))
        log_test = sum(w * np.log(np.maximum(t, eps)) for w, t in zip(weights, tests))
        oof_blend = np.exp(log_oof)
        test_blend = np.exp(log_test)
    elif agg == "rank_mean":
        oof_blend = sum(w * _rank_within_each(o) for w, o in zip(weights, oofs))
        test_blend = sum(w * _rank_within_each(t) for w, t in zip(weights, tests))
    else:
        raise ValueError(f"unknown blend.aggregation: {agg}")

    if bcfg.get("clip_to_unit"):
        oof_blend = np.clip(oof_blend, 0, 1)
        test_blend = np.clip(test_blend, 0, 1)

    info = {"mode": "blend", "aggregation": agg, "weights": weights}
    return oof_blend, test_blend, info


def _stack(
    oofs: list[np.ndarray],
    tests: list[np.ndarray],
    y_true: np.ndarray,
    cv_indices: list[tuple[np.ndarray, np.ndarray]],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a meta-learner on OOFs, predict OOF (via CV) + test."""
    scfg = cfg["stack"] or {}
    model_name = scfg.get("meta_model", "ridge")

    # Stack features: each source contributes its OOF as a column (or columns for multiclass).
    def to_2d(a: np.ndarray) -> np.ndarray:
        return a.reshape(-1, 1) if a.ndim == 1 else a

    X_oof = np.hstack([to_2d(o) for o in oofs])
    X_test = np.hstack([to_2d(t) for t in tests])
    n_train = X_oof.shape[0]
    out_oof = np.zeros(n_train, dtype=np.float64)
    out_test_acc = np.zeros(X_test.shape[0], dtype=np.float64)

    if model_name == "ridge":
        from sklearn.linear_model import Ridge  # noqa: PLC0415

        alpha = float(scfg.get("ridge_alpha", 1.0))
        for tr_idx, va_idx in cv_indices:
            m = Ridge(alpha=alpha)
            m.fit(X_oof[tr_idx], y_true[tr_idx])
            out_oof[va_idx] = m.predict(X_oof[va_idx])
            out_test_acc += m.predict(X_test) / len(cv_indices)
        coefs = m.coef_.tolist()  # from last fold; informational
        info = {"mode": "stack", "meta_model": "ridge", "alpha": alpha, "coef_last_fold": coefs}
    elif model_name == "lgbm":
        import lightgbm as lgb  # noqa: PLC0415

        params = scfg.get("lgbm_params", {})
        n_estimators = int(params.pop("n_estimators", 500))
        for tr_idx, va_idx in cv_indices:
            m = lgb.LGBMRegressor(**params, n_estimators=n_estimators)
            m.fit(
                X_oof[tr_idx],
                y_true[tr_idx],
                eval_set=[(X_oof[va_idx], y_true[va_idx])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            out_oof[va_idx] = m.predict(X_oof[va_idx])
            out_test_acc += m.predict(X_test) / len(cv_indices)
        info = {"mode": "stack", "meta_model": "lgbm", "n_estimators": n_estimators}
    else:
        raise ValueError(f"unknown meta_model: {model_name}")

    return out_oof, out_test_acc, info


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Mirror tabular-lgbm's metric function to keep behavior identical.
    sys.path.insert(0, str(Path(__file__).parent.parent / "tabular-lgbm"))
    from train import _compute_metric as _cm  # noqa: PLC0415

    return _cm(metric_name, y_true, y_pred)


def _cv_indices(cfg: dict[str, Any], y_true: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    sys.path.insert(0, str(Path(__file__).parent.parent / "tabular-lgbm"))
    from train import _split_iter  # noqa: PLC0415

    # Build a dummy DataFrame with just the target so KFold/StratifiedKFold get the shape.
    dummy = pd.DataFrame({"target": y_true})
    return [(tr, va) for _, tr, va in _split_iter(cfg, dummy, y_true)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--progress-log", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    progress_log = Path(args.progress_log)
    cfg = yaml.safe_load((run_dir / args.config).read_text())

    source_runs = cfg.get("source_runs") or []
    if not source_runs:
        sys.stderr.write("ensemble: source_runs is empty; nothing to blend.\n")
        return 2

    t0 = time.time()
    _append_progress(
        progress_log,
        {"stage": "stage2", "event": "ensemble_started", "run_id": cfg["run_id"],
         "mode": cfg["mode"], "sources": source_runs},
    )

    oofs: list[np.ndarray] = []
    tests: list[np.ndarray] = []
    test_sub_columns: list[str] | None = None
    sample_sub = pd.read_csv(cfg["data_paths"]["sample_submission"])
    id_col_lower = (cfg.get("feature_config") or {}).get("id_col", "id").lower()
    test_sub_columns = [c for c in sample_sub.columns if c.lower() != id_col_lower]

    for src_id in source_runs:
        oof, preds_df, _metric_name, _cv_score = _load_source(run_dir, src_id)
        oofs.append(oof)
        # Extract test predictions as array in same column order as sample_sub (minus id).
        # Tolerate single-col or multi-col.
        cols = [c for c in preds_df.columns if c.lower() != id_col_lower]
        arr = preds_df[cols].to_numpy(dtype=np.float64)
        if arr.shape[1] == 1:
            arr = arr.ravel()
        tests.append(arr)

    train_df = pd.read_csv(cfg["data_paths"]["train"])
    target_col = (cfg.get("feature_config") or {}).get("target_col", "target")
    y_true = train_df[target_col].to_numpy()
    metric_name = (cfg.get("metric") or {}).get("name", "rmse")

    # Shape sanity
    n_train = len(train_df)
    for i, o in enumerate(oofs):
        if len(o) != n_train:
            sys.stderr.write(
                f"source {source_runs[i]} oof shape {o.shape} mismatched n_train={n_train}\n"
            )
            return 2

    if cfg["mode"] == "blend":
        oof_pred, test_pred, info = _blend(oofs, tests, cfg)
    elif cfg["mode"] == "stack":
        cv_idx = _cv_indices(cfg, y_true)
        oof_pred, test_pred, info = _stack(oofs, tests, y_true, cv_idx, cfg)
    else:
        sys.stderr.write(f"unknown mode: {cfg['mode']}\n")
        return 2

    # Per-fold CV scores via the same CV indices (so cv_std is meaningful even for blends).
    cv_idx_full = _cv_indices(cfg, y_true)
    per_fold = []
    for _tr_idx, va_idx in cv_idx_full:
        per_fold.append(_compute_metric(metric_name, y_true[va_idx], oof_pred[va_idx] if oof_pred.ndim == 1 else oof_pred[va_idx]))
    cv_mean = float(np.mean(per_fold)) if per_fold else 0.0
    cv_std = float(np.std(per_fold)) if per_fold else 0.0

    _atomic_save_npy(run_dir / "oof.npy", oof_pred)

    # Write test_preds in submission format.
    sub = sample_sub.copy()
    if test_pred.ndim == 1 and len(test_sub_columns) == 1:
        sub[test_sub_columns[0]] = test_pred
    else:
        for k, c in enumerate(test_sub_columns):
            sub[c] = test_pred[:, k] if test_pred.ndim > 1 else test_pred
    _atomic_save_csv(run_dir / "test_preds.csv", sub)

    # cv_score.json
    (run_dir / "cv_score.json").write_text(
        json.dumps(
            {
                "metric": metric_name,
                "direction": (cfg.get("metric") or {}).get("direction", "maximize"),
                "per_fold": per_fold,
                "mean": cv_mean,
                "std": cv_std,
                "n_train": n_train,
                "n_test": len(sample_sub),
                "seeds": cfg.get("seeds", [42]),
                "n_folds": int(cfg["cv_split"]["n_folds"]),
                "ensemble": info,
                "source_runs": source_runs,
            },
            indent=2,
        )
    )

    # Append a "Construction" section to attribution.md (modeling.py wrote the base).
    attr_path = run_dir / "attribution.md"
    construction = (
        "\n## Ensemble construction\n\n"
        f"- Mode: `{cfg['mode']}`\n"
        f"- Info: `{json.dumps(info)}`\n"
        f"- Sources: {', '.join(f'`{s}`' for s in source_runs)}\n"
        f"- Combined CV: {metric_name} = {cv_mean:.6f} (std {cv_std:.6f})\n"
    )
    if attr_path.exists():
        with attr_path.open("a") as f:
            f.write(construction)
    else:
        attr_path.write_text(construction)

    _append_progress(
        progress_log,
        {"stage": "stage2", "event": "ensemble_done", "run_id": cfg["run_id"],
         "cv_score": cv_mean, "cv_std": cv_std, "wallclock_s": time.time() - t0},
    )
    print(f"ensemble CV {metric_name} = {cv_mean:.6f} (std {cv_std:.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
