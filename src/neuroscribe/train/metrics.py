"""Evaluation metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import r2_score


def _compute_basic(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    metrics["r2"] = r2_score(y_true_flat, y_pred_flat, multioutput="variance_weighted")
    corrs = []
    for d in range(y_true_flat.shape[-1]):
        yt = y_true_flat[:, d]
        yp = y_pred_flat[:, d]
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            corrs.append(0.0)
        else:
            corrs.append(np.corrcoef(yt, yp)[0, 1])
    metrics["corr"] = float(np.mean(corrs))

    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    metrics["rmse"] = float(np.sqrt(mse))
    return metrics


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, aggregate: str = "micro"
) -> Dict[str, float]:
    if aggregate == "micro" or y_true.ndim < 3:
        return _compute_basic(y_true, y_pred)
    if aggregate != "macro":
        raise ValueError(f"Unknown metrics aggregation '{aggregate}'")

    r2_vals = []
    corr_vals = []
    rmse_vals = []
    for i in range(y_true.shape[0]):
        metrics = _compute_basic(y_true[i], y_pred[i])
        r2_vals.append(metrics["r2"])
        corr_vals.append(metrics["corr"])
        rmse_vals.append(metrics["rmse"])
    return {
        "r2": float(np.nanmean(r2_vals)),
        "corr": float(np.nanmean(corr_vals)),
        "rmse": float(np.nanmean(rmse_vals)),
    }
