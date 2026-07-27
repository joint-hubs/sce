"""
@module: tests.equity.test_schema
@depends: equity.data.schema, pandera, pandas
@exports:
@data_flow: synthetic frames -> validate_prices / assert_primary_key_unique
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity.data.schema import (
    CANONICAL_PRICE_COLUMNS,
    assert_primary_key_unique,
    prices_schema,
    validate_prices,
)


def _valid_frame(n: int = 3) -> pd.DataFrame:
    """Return a small valid prices frame (tz-aware NY session closes)."""
    dates = pd.bdate_range("2024-01-02", periods=n)
    close_ts = dates + pd.Timedelta(hours=16)  # naive 16:00
    close_ts = close_ts.tz_localize("America/New_York")  # tz-aware 16:00 ET
    base = 100.0
    return pd.DataFrame(
        {
            "ticker": "AAPL",
            "period_close_ts": close_ts,
            "open": base - 1.0,
            "high": base + 1.0,
            "low": base - 2.0,
            "close": base,
            "adj_close": base,
            "volume": 1_000_000.0,
            "hlc_average": (base + 1.0 + base - 2.0 + base) / 3.0,
        }
    )


def test_validate_prices_passes_valid_frame():
    df = _valid_frame()
    out = validate_prices(df)
    assert list(out.columns) == CANONICAL_PRICE_COLUMNS
    assert out["period_close_ts"].dt.tz is not None


def test_validate_prices_missing_column_fails():
    df = _valid_frame().drop(columns=["hlc_average"])
    with pytest.raises(Exception, match="hlc_average"):
        validate_prices(df)


def test_validate_prices_extra_column_fails_strict():
    df = _valid_frame()
    df["unexpected"] = 1
    with pytest.raises(Exception):
        validate_prices(df)


def test_validate_prices_tz_naive_period_close_ts_fails():
    """The tz guard: tz-naive period_close_ts must be rejected."""
    df = _valid_frame()
    naive = df["period_close_ts"].dt.tz_localize(None)
    df = df.assign(period_close_ts=naive)
    with pytest.raises(Exception, match="America/New_York"):
        validate_prices(df)


def test_validate_prices_coerces_floats_and_keeps_tz():
    """Integer OHLCV fields should be coerced to float; tz is preserved."""
    df = _valid_frame()
    # Pass integers for OHLCV; schema should coerce to float.
    for col in ("open", "high", "low", "close", "adj_close", "volume", "hlc_average"):
        df[col] = df[col].astype(int)
    out = validate_prices(df)
    for col in ("open", "high", "low", "close", "adj_close", "volume", "hlc_average"):
        assert pd.api.types.is_float_dtype(out[col]), f"{col} not float after coerce"
    assert out["period_close_ts"].dt.tz is not None


def test_assert_primary_key_unique_passes():
    df = _valid_frame(n=5)
    assert_primary_key_unique(df)  # no exception


def test_assert_primary_key_unique_detects_duplicates():
    df = _valid_frame(n=3)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        assert_primary_key_unique(dup)


def test_assert_primary_key_unique_noop_on_empty():
    empty = pd.DataFrame(columns=CANONICAL_PRICE_COLUMNS)
    assert_primary_key_unique(empty)  # no exception


def test_prices_schema_is_strict_and_tz_aware_dtype():
    # Sanity: the schema object is configured as expected.
    assert prices_schema.strict is True
    # The period_close_ts column should be declared with a tz-aware dtype.
    # pandera wraps the dtype in its own DataType; assert via the str form.
    col = prices_schema.columns["period_close_ts"]
    dtype_str = str(col.dtype)
    assert "America/New_York" in dtype_str, f"unexpected dtype: {dtype_str}"


def test_nullable_ohlcv_with_nan_passes():
    """Partial bars (NaN OHLCV) for delisted-within-window tickers pass."""
    df = _valid_frame(n=2)
    df.loc[0, ["open", "high", "low", "close", "adj_close", "volume", "hlc_average"]] = np.nan
    out = validate_prices(df)
    assert out["close"].isna().any()
