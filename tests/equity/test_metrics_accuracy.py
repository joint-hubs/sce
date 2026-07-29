"""
@module: tests.equity.test_metrics_accuracy
@depends: equity.metrics.accuracy
@exports:
@data_flow: hand fixtures -> rmse/mae/hit_rate/aggregate_accuracy

S6.2 accuracy metric unit tests. Exact values on small hand-computed fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from equity.metrics.accuracy import (
    aggregate_accuracy,
    directional_hit_rate,
    mae,
    rmse,
)


def test_rmse_exact_four_point() -> None:
    # errors: 0, 2, -2, 1 → mse = (0+4+4+1)/4 = 2.25 → rmse = 1.5
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 4.0, 1.0, 5.0]
    assert rmse(y_true, y_pred) == pytest.approx(1.5)


def test_mae_exact_four_point() -> None:
    # abs errors: 0, 2, 2, 1 → mae = 5/4 = 1.25
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 4.0, 1.0, 5.0]
    assert mae(y_true, y_pred) == pytest.approx(1.25)


def test_directional_hit_rate_known_mix() -> None:
    # signs true: + + - 0 - ; pred: + - - + - → hits on idx 0,2,4 = 3/5
    y_true = np.array([0.5, 0.2, -0.3, 0.0, -1.0])
    y_pred = np.array([0.1, -0.1, -0.2, 0.4, -0.5])
    assert directional_hit_rate(y_true, y_pred) == pytest.approx(0.6)


def test_nan_pairs_dropped() -> None:
    y_true = [1.0, np.nan, 3.0, 4.0]
    y_pred = [1.0, 2.0, np.nan, 6.0]
    # only first and last pairs finite: errors 0 and 2 → rmse=sqrt(2), mae=1
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(2.0))
    assert mae(y_true, y_pred) == pytest.approx(1.0)
    assert directional_hit_rate(y_true, y_pred) == pytest.approx(1.0)


def test_all_nan_returns_nan() -> None:
    y_true = [np.nan, np.nan]
    y_pred = [1.0, 2.0]
    assert math.isnan(rmse(y_true, y_pred))
    assert math.isnan(mae(y_true, y_pred))
    assert math.isnan(directional_hit_rate(y_true, y_pred))


def test_aggregate_accuracy_mean_std_across_folds() -> None:
    # two folds, one horizon; fold0 perfect, fold1 constant error 2 on two pts
    fold0 = {
        1: pd.DataFrame(
            {
                "ret_h1": [0.1, -0.2, 0.3],
                "pred_h1": [0.1, -0.2, 0.3],
                "split": ["test"] * 3,
            }
        )
    }
    fold1 = {
        1: pd.DataFrame(
            {
                "ret_h1": [0.0, 1.0],
                "pred_h1": [2.0, 3.0],  # abs err 2,2 → mae=2, rmse=2
                "split": ["test"] * 2,
            }
        )
    }
    out = aggregate_accuracy([fold0, fold1], horizons=(1,))
    assert 1 in out
    m = out[1]
    assert m["rmse_mean"] == pytest.approx((0.0 + 2.0) / 2.0)
    assert m["mae_mean"] == pytest.approx((0.0 + 2.0) / 2.0)
    # sample std of [0, 2] = sqrt(2)
    assert m["rmse_std"] == pytest.approx(math.sqrt(2.0))
    assert m["mae_std"] == pytest.approx(math.sqrt(2.0))
    # hit rates: fold0=1.0 ; fold1 signs (0 vs +, + vs +) → hit only on second = 0.5
    assert m["hit_rate_mean"] == pytest.approx(0.75)
