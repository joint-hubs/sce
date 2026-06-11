"""
@module: tests.test_prepare_m5_dataset
@depends: scripts.prepare_m5_dataset
@exports:
@data_flow: raw M5-like frames -> aggregated benchmark panel
"""

from __future__ import annotations

import pandas as pd

from scripts.prepare_m5_dataset import build_store_dept_panel


def test_build_store_dept_panel_creates_temporal_features():
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    calendar_df = pd.DataFrame(
        {
            "d": [f"d_{i + 1}" for i in range(35)],
            "date": dates,
            "wm_yr_wk": [1] * 7 + [2] * 7 + [3] * 7 + [4] * 7 + [5] * 7,
            "wday": [(i % 7) + 1 for i in range(35)],
            "month": [1] * 35,
            "year": [2024] * 35,
            "event_name_1": [None] * 35,
            "event_type_1": [None] * 35,
            "event_name_2": [None] * 35,
            "event_type_2": [None] * 35,
            "snap_CA": [0] * 35,
            "snap_TX": [1] * 35,
            "snap_WI": [0] * 35,
        }
    )
    sales_df = pd.DataFrame(
        {
            "item_id": ["item_1"],
            "dept_id": ["dept_1"],
            "cat_id": ["cat_1"],
            "store_id": ["store_1"],
            "state_id": ["TX"],
            **{f"d_{i + 1}": [i + 1] for i in range(35)},
        }
    )
    sell_prices_df = pd.DataFrame(
        {
            "store_id": ["store_1"] * 5,
            "item_id": ["item_1"] * 5,
            "wm_yr_wk": [1, 2, 3, 4, 5],
            "sell_price": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )

    panel = build_store_dept_panel(calendar_df, sales_df, sell_prices_df, history_days=35)

    assert not panel.empty
    assert "lag_7" in panel.columns
    assert "rolling_mean_28" in panel.columns
    assert panel["date"].min() == pd.Timestamp("2024-01-29")
    assert panel.iloc[0]["snap_state"] == 1