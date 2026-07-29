"""
@module: equity.metrics.accuracy
@depends: numpy, pandas
@exports: rmse, mae, directional_hit_rate, aggregate_accuracy
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §8 (S6.2 accuracy)
@data_flow: y_true/y_pred arrays or per-fold test prediction frames
            -> scalar accuracy / per-horizon mean±std across folds

NaN-safe point-forecast accuracy helpers used by the walk-forward runner (S6.2).
Sharpe / Sortino / decile portfolio metrics live in a later slice (S6.3).
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


def _paired_finite(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Drop pairs where either side is non-finite; return 1-d float arrays."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true/y_pred shape mismatch: {yt.shape} vs {yp.shape}"
        )
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root-mean-square error over finite pairs. NaN if no finite pairs remain."""
    yt, yp = _paired_finite(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error over finite pairs. NaN if no finite pairs remain."""
    yt, yp = _paired_finite(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def directional_hit_rate(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of finite pairs where ``sign(y_pred) == sign(y_true)``.

    Zero is treated as its own sign (``np.sign(0) == 0``), so a zero true return
    only hits when the prediction is also zero. Returns NaN if no finite pairs.
    """
    yt, yp = _paired_finite(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.sign(yt) == np.sign(yp)))


def aggregate_accuracy(
    per_fold_test_predictions: Sequence[Mapping[int, pd.DataFrame]],
    horizons: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """Aggregate per-horizon accuracy mean±std **across folds** on the TEST slice.

    Parameters
    ----------
    per_fold_test_predictions:
        Sequence (one entry per fold) of ``{horizon: DataFrame}`` where each frame
        carries at least ``ret_h{N}`` and ``pred_h{N}`` for that horizon. Frames
        may also carry a ``split`` column; when present only ``split == "test"``
        rows are used (defensive — callers should already pass test-only frames).
    horizons:
        Horizons to score.

    Returns
    -------
    dict
        ``{h: {rmse_mean, rmse_std, mae_mean, mae_std, hit_rate_mean, hit_rate_std}}``.
        Std is sample std (ddof=1) when >=2 folds, else 0.0. Empty-fold NaNs are
        dropped before mean/std; if no fold yields a finite score the mean/std
        are NaN.
    """
    out: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        h = int(h)
        y_col = f"ret_h{h}"
        p_col = f"pred_h{h}"
        rmses: list[float] = []
        maes: list[float] = []
        hits: list[float] = []
        for fold_map in per_fold_test_predictions:
            if h not in fold_map:
                continue
            frame = fold_map[h]
            if frame is None or len(frame) == 0:
                continue
            if "split" in frame.columns:
                frame = frame.loc[frame["split"].astype(str) == "test"]
            if y_col not in frame.columns or p_col not in frame.columns:
                continue
            r = rmse(frame[y_col], frame[p_col])
            m = mae(frame[y_col], frame[p_col])
            hit = directional_hit_rate(frame[y_col], frame[p_col])
            if np.isfinite(r):
                rmses.append(float(r))
            if np.isfinite(m):
                maes.append(float(m))
            if np.isfinite(hit):
                hits.append(float(hit))

        def _mean_std(vals: list[float]) -> tuple[float, float]:
            if not vals:
                return float("nan"), float("nan")
            arr = np.asarray(vals, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
            return mean, std

        rmse_mean, rmse_std = _mean_std(rmses)
        mae_mean, mae_std = _mean_std(maes)
        hit_mean, hit_std = _mean_std(hits)
        out[h] = {
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "hit_rate_mean": hit_mean,
            "hit_rate_std": hit_std,
        }
    return out
