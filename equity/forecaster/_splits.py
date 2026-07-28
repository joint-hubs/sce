"""
@module: equity.forecaster._splits
@depends: numpy, pandas, sklearn
@exports: ts_group_split, rolling_ts_folds, make_design_matrix, FEATURE_BLOCKLIST_PREFIXES
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §9.3 / FOC-51 R1 ts-group
@data_flow: frame[time_col] -> unique timestamps -> train/test or rolling OOF folds

Internal helpers shared by the sector / residual / quantile heads.

**ts-group split (P0):** never row-iloc. A boundary day must not be cut mid
cross-section (FOC-51 R1). Mirrors ``equity.diagnostics.sce_reuse`` lines 204-213.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Columns whose name starts with any of these MUST never enter a design matrix
# as a feature. Predictions (``pred_*``) are also blocked;} residual injects
# ``pred_sector_hN`` explicitly after the design-matrix build.
FEATURE_BLOCKLIST_PREFIXES: Tuple[str, ...] = (
    "ret_h",  # forward targets
    "pred_h",  # final / residual preds
    "pred_sector_h",  # sector-head preds (injected only where intentional)
    "pred_resid_h",  # residual-layer preds
    "resid_h",  # residual labels
)


def ts_group_split(
    frame: pd.DataFrame,
    *,
    time_col: str = "period_close_ts",
    test_frac: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single ts-group train/test split (smoke-run geometry).

    Mirrors ``equity.diagnostics.sce_reuse`` evaluate path: split on sorted unique
    timestamps so a boundary day is never cut mid cross-section.
    """
    if not 0.0 < float(test_frac) < 1.0:
        raise ValueError(f"test_frac must be in (0, 1); got {test_frac}")
    if time_col not in frame.columns:
        raise ValueError(f"ts_group_split: missing time_col {time_col!r}")

    sort_keys = [time_col] + (["ticker"] if "ticker" in frame.columns else [])
    ordered = frame.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    unique_ts = pd.Index(ordered[time_col].drop_duplicates().sort_values())
    n_ts = len(unique_ts)
    if n_ts < 2:
        raise ValueError(f"ts_group_split: need >= 2 distinct timestamps; got {n_ts}")
    split_idx = max(1, min(n_ts - 1, int(round(n_ts * (1.0 - float(test_frac))))))
    split_day = unique_ts[split_idx]
    train = ordered.loc[ordered[time_col] < split_day].copy()
    test = ordered.loc[ordered[time_col] >= split_day].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"ts_group_split: empty train/test after timestamp-group split "
            f"(n_ts={n_ts}, split_idx={split_idx}, split_day={split_day}, "
            f"test_frac={test_frac})."
        )
    return train, test


def rolling_ts_folds(
    frame: pd.DataFrame,
    *,
    time_col: str = "period_close_ts",
    n_folds: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Reproduce SCE rolling TimeSeriesSplit geometry on **unique timestamps**.

    Returns a list of ``(train_row_positions, val_row_positions)`` into the
    **time-sorted** ``frame.reset_index(drop=True)``. Positions are integer
    locations suitable for ``.iloc``.

    Geometry mirrors ``sce.engine`` rolling CF (TimeSeriesSplit on the ordered
    axis) but the split unit is a timestamp group, never a half-cut day.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2; got {n_folds}")
    if time_col not in frame.columns:
        raise ValueError(f"rolling_ts_folds: missing time_col {time_col!r}")

    sort_keys = [time_col] + (["ticker"] if "ticker" in frame.columns else [])
    ordered = frame.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    unique_ts = pd.Index(ordered[time_col].drop_duplicates().sort_values())
    n_ts = len(unique_ts)
    if n_ts < n_folds + 1:
        raise ValueError(
            f"rolling_ts_folds: need >= n_folds+1 distinct timestamps "
            f"(n_folds={n_folds}); got n_ts={n_ts}."
        )

    # TimeSeriesSplit on the timestamp axis (not on rows). Map each test/train
    # timestamp index set back to all row positions sharing those timestamps.
    max_feasible_test = max(1, n_ts // (n_folds + 1))
    max_train_size = max(2 * max_feasible_test, int(n_ts * 0.8))
    splitter = TimeSeriesSplit(
        n_splits=n_folds,
        max_train_size=max_train_size,
        test_size=max_feasible_test,
    )
    # Dummy X of length n_ts.
    dummy = np.zeros(n_ts)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    ts_to_rows: dict[object, np.ndarray] = {
        ts: ordered.index[ordered[time_col] == ts].to_numpy() for ts in unique_ts
    }

    for train_ts_idx, val_ts_idx in splitter.split(dummy):
        train_rows = np.concatenate([ts_to_rows[unique_ts[i]] for i in train_ts_idx])
        val_rows = np.concatenate([ts_to_rows[unique_ts[i]] for i in val_ts_idx])
        train_rows.sort()
        val_rows.sort()
        folds.append((train_rows, val_rows))
    return folds


def is_blocked_feature(col: str) -> bool:
    """True if ``col`` must NEVER enter a default design matrix."""
    if not isinstance(col, str):
        return True
    return any(col == p or col.startswith(p) for p in FEATURE_BLOCKLIST_PREFIXES)


def make_design_matrix(
    frame: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    extra_cols: Optional[Sequence[str]] = None,
    sector_col: str = "sector",
    cast_sector_category: bool = True,
    exclude: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build an X matrix that never contains blocked forward/residual columns.

    Default feature selection: all numeric columns that are not blocked, plus
    ``sector`` (cast to categorical dtype for XGBoost native categoricals) when
    present. ``extra_cols`` is appended after the blocklist filter (used by the
    residual layer to inject the **OOF** ``pred_sector_hN`` feature intentionally).
    """
    exclude_set = set(exclude or ())
    if feature_cols is None:
        cols: List[str] = []
        for c in frame.columns:
            if c in exclude_set or is_blocked_feature(c):
                continue
            if c == sector_col:
                cols.append(c)
                continue
            if pd.api.types.is_numeric_dtype(frame[c]):
                cols.append(c)
    else:
        cols = [c for c in feature_cols if c in frame.columns and c not in exclude_set]
        cols = [c for c in cols if not is_blocked_feature(c)]

    if extra_cols:
        for c in extra_cols:
            if c not in cols and c in frame.columns:
                cols.append(c)

    if not cols:
        raise ValueError("make_design_matrix: no feature columns selected")

    X = frame.loc[:, cols].copy()
    if cast_sector_category and sector_col in X.columns:
        X[sector_col] = X[sector_col].astype("category")
    return X, cols
