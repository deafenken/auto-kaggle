"""Shared CV + metric helpers used by every training template.

This file lives at `auto-kaggle-modeling/assets/templates/_cv_common.py`. The
dispatcher (`modeling.py`) copies it into every run directory alongside the
chosen template's `train.py`, so each template imports it as a local sibling
module (`from _cv_common import _split_iter, _compute_metric`).

Why a shared module instead of duplication: the prior layout had templates
importing from `../tabular-lgbm/train.py`, which works inside the source tree
but not after a template is copied into `runs/<comp>/.../runs/<run_id>/` —
that sibling doesn't exist at runtime. Codex review caught it.

Implements the full CV scheme set documented in
`auto-kaggle-modeling/references/cv-design.md`:

- StratifiedKFold (binned regression target supported via `stratify_col` virtual)
- KFold
- GroupKFold
- StratifiedGroupKFold (sklearn >= 1.0)
- TimeSeriesSplit (with optional `gap` and `test_size_days`)
- blocked-time (custom block lists)

And the metric helpers cover the canonical set in
`auto-kaggle-bootstrap/references/task-type-detection.md` (RMSE, RMSLE, MAE,
AUC, log_loss, accuracy, F1, plus IoU / Dice for seg templates).
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# CV scheme dispatch
# ----------------------------------------------------------------------------


def _split_iter(
    cfg: dict, df: pd.DataFrame, y: np.ndarray
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield `(fold_idx, train_idx, val_idx)` per cv_split.yaml.

    The cfg argument is the run's full config dict; the CV portion lives under
    `cfg["cv_split"]`. See `auto-kaggle-modeling/references/cv-design.md` for the
    canonical schema of each scheme.
    """
    cv = cfg["cv_split"]
    scheme = cv["scheme"]
    n_folds = int(cv["n_folds"])
    seed = cv.get("seed")
    shuffle = bool(cv.get("shuffle", True))
    group_col = cv.get("group_col")
    stratify_col = cv.get("stratify_col")

    if scheme == "StratifiedKFold":
        from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

        # Optional binning of a continuous target for stratified regression CV.
        y_strat = _stratify_target(df, y, stratify_col, n_bins=int(cv.get("stratify_bins", 10)))
        sk = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed if shuffle else None)
        for i, (tr, va) in enumerate(sk.split(df, y_strat)):
            yield i, tr, va
        return

    if scheme == "KFold":
        from sklearn.model_selection import KFold  # noqa: PLC0415

        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed if shuffle else None)
        for i, (tr, va) in enumerate(kf.split(df)):
            yield i, tr, va
        return

    if scheme == "GroupKFold":
        from sklearn.model_selection import GroupKFold  # noqa: PLC0415

        if group_col is None:
            raise ValueError("GroupKFold requires `cv_split.group_col`")
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not in train columns")
        gk = GroupKFold(n_splits=n_folds)
        for i, (tr, va) in enumerate(gk.split(df, y, groups=df[group_col].to_numpy())):
            yield i, tr, va
        return

    if scheme == "StratifiedGroupKFold":
        # sklearn >= 1.0
        from sklearn.model_selection import StratifiedGroupKFold  # noqa: PLC0415

        if group_col is None:
            raise ValueError("StratifiedGroupKFold requires `cv_split.group_col`")
        y_strat = _stratify_target(df, y, stratify_col, n_bins=int(cv.get("stratify_bins", 10)))
        sgk = StratifiedGroupKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed if shuffle else None)
        for i, (tr, va) in enumerate(sgk.split(df, y_strat, groups=df[group_col].to_numpy())):
            yield i, tr, va
        return

    if scheme == "TimeSeriesSplit":
        from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415

        gap = int(cv.get("gap", 0))
        test_size = cv.get("test_size", None)
        kwargs = {"n_splits": n_folds, "gap": gap}
        if test_size is not None:
            kwargs["test_size"] = int(test_size)
        ts = TimeSeriesSplit(**kwargs)
        for i, (tr, va) in enumerate(ts.split(df)):
            yield i, tr, va
        return

    if scheme == "blocked-time":
        # Custom blocks: cv_split.blocks is a list of {train: [...], val: [...]}
        # values; we look up rows in df by `cv_split.block_col` (str-compared).
        blocks = cv.get("blocks") or []
        block_col = cv.get("block_col", "year")
        if not blocks:
            raise ValueError("blocked-time CV requires a non-empty cv_split.blocks list")
        if block_col not in df.columns:
            raise ValueError(f"block_col '{block_col}' not in train columns")
        col_values = df[block_col].astype(str).to_numpy()
        for i, block in enumerate(blocks):
            train_vals = {str(v) for v in block.get("train", [])}
            val_vals = {str(v) for v in block.get("val", [])}
            tr = np.where(np.isin(col_values, list(train_vals)))[0]
            va = np.where(np.isin(col_values, list(val_vals)))[0]
            if len(tr) == 0 or len(va) == 0:
                raise ValueError(
                    f"blocked-time fold {i} is empty: train={len(tr)} val={len(va)}"
                )
            yield i, tr, va
        return

    raise ValueError(
        f"unsupported CV scheme '{scheme}' — supported: StratifiedKFold | KFold | "
        "GroupKFold | StratifiedGroupKFold | TimeSeriesSplit | blocked-time"
    )


def _stratify_target(
    df: pd.DataFrame, y: np.ndarray, stratify_col: Optional[str], n_bins: int = 10
) -> np.ndarray:
    """Return the labels to stratify on.

    - If `stratify_col` is set and exists in df, use that column.
    - Otherwise, if y looks regression-y (continuous floats), bin into quantiles.
    - Otherwise, return y unchanged (classification target).
    """
    if stratify_col and stratify_col in df.columns:
        return df[stratify_col].to_numpy()
    if y.dtype.kind == "f":
        # Continuous; bin for stratification.
        try:
            bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
            return np.asarray(bins)
        except (ValueError, TypeError):
            return y
    return y


# ----------------------------------------------------------------------------
# Metric dispatch
# ----------------------------------------------------------------------------


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score `y_pred` against `y_true` using the named competition metric.

    Coverage matches the metric set in
    `auto-kaggle-bootstrap/references/task-type-detection.md`. For multi-class,
    pass class probabilities (shape `(n, n_classes)`) and use AUC / log_loss /
    accuracy; for binary, pass probabilities of the positive class.
    """
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        cohen_kappa_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        roc_auc_score,
    )

    m = (metric_name or "").lower()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if m == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred.reshape(y_true.shape))))
    if m == "rmsle":
        p = np.clip(y_pred, 0, None).reshape(y_true.shape)
        return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(p))))
    if m == "mae":
        return float(mean_absolute_error(y_true, y_pred.reshape(y_true.shape)))
    if m == "auc":
        if y_pred.ndim == 2 and y_pred.shape[1] > 2:
            return float(roc_auc_score(y_true, y_pred, multi_class="ovr"))
        return float(roc_auc_score(y_true, y_pred.reshape(-1) if y_pred.ndim == 2 else y_pred))
    if m in ("log_loss", "binary_logloss", "multi_logloss"):
        return float(log_loss(y_true, y_pred))
    if m == "accuracy":
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            pred_cls = np.argmax(y_pred, axis=-1)
        else:
            pred_cls = (y_pred > 0.5).astype(int)
        return float(accuracy_score(y_true, pred_cls))
    if m == "f1":
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            pred_cls = np.argmax(y_pred, axis=-1)
        else:
            pred_cls = (y_pred > 0.5).astype(int)
        return float(f1_score(y_true, pred_cls, average="macro"))
    if m in ("quadratic_weighted_kappa", "qwk"):
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            pred_cls = np.argmax(y_pred, axis=-1)
        else:
            pred_cls = np.clip(np.round(y_pred), y_true.min(), y_true.max()).astype(int)
        return float(cohen_kappa_score(y_true, pred_cls, weights="quadratic"))
    if m in ("iou", "jaccard"):
        return _iou_score(y_true, y_pred)
    if m == "dice":
        return _dice_score(y_true, y_pred)
    # Fallback: warn loudly via the exception so the agent escalates rather than
    # silently submitting on the wrong metric.
    raise ValueError(
        f"metric '{metric_name}' not implemented in _cv_common._compute_metric. "
        "Add it here or compute it in the run-dir copy of train.py."
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
