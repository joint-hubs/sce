"""
@module: scripts.prepare_m5_dataset
@depends:
@exports: build_store_dept_panel, main
@data_flow: Kaggle raw CSVs -> aggregated temporal parquet

Prepare an M5-derived benchmark for SCE.

Usage:
    python scripts/prepare_m5_dataset.py --download
    python scripts/prepare_m5_dataset.py --raw-dir data/raw/m5 --output data/parquet/m5_store_dept_daily.parquet
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "m5"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "parquet" / "m5_store_dept_daily.parquet"
COMPETITION = "m5-forecasting-accuracy"
REQUIRED_FILES = ["calendar.csv", "sales_train_validation.csv", "sell_prices.csv"]


def _resolve_kaggle_command() -> list[str]:
    for name in ("kaggle.exe", "kaggle"):
        candidate = Path(sys.executable).with_name(name)
        if candidate.exists():
            return [str(candidate)]
    return [sys.executable, "-m", "kaggle.cli"]


def _extract_zip(archive_path: Path, output_dir: Path) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_dir)
        members = [name for name in archive.namelist() if not name.endswith("/")]
    if len(members) != 1:
        raise RuntimeError(f"Expected one file inside {archive_path.name}, found {members}")
    return output_dir / members[0]


def _download_kaggle_file(file_name: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = subprocess.run(
            [
                *_resolve_kaggle_command(),
                "competitions",
                "download",
                "-c",
                COMPETITION,
                "-f",
                file_name,
                "-p",
                tmp_dir,
                "-q",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error_text = result.stderr.strip() or result.stdout.strip() or "unknown Kaggle error"
            raise RuntimeError(error_text)

        zip_path = Path(tmp_dir) / f"{file_name}.zip"
        extracted_path = _extract_zip(zip_path, Path(tmp_dir)) if zip_path.exists() else Path(tmp_dir) / file_name
        if not extracted_path.exists():
            raise RuntimeError(f"Downloaded file not found for {file_name}")

        output_path = raw_dir / file_name
        shutil.move(str(extracted_path), output_path)
        return output_path


def ensure_m5_raw_files(raw_dir: Path, download_missing: bool) -> dict[str, Path]:
    paths = {name: raw_dir / name for name in REQUIRED_FILES}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing and not download_missing:
        raise FileNotFoundError(
            "Missing M5 raw files: "
            + ", ".join(missing)
            + ". Re-run with --download after configuring Kaggle credentials."
        )

    for name in missing:
        _download_kaggle_file(name, raw_dir)
    return paths


def build_store_dept_panel(
    calendar_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    sell_prices_df: pd.DataFrame,
    history_days: int = 365,
) -> pd.DataFrame:
    sales_df = sales_df.copy()
    calendar_df = calendar_df.copy()
    sell_prices_df = sell_prices_df.copy()

    value_cols = [col for col in sales_df.columns if col.startswith("d_")]
    if history_days > 0:
        value_cols = value_cols[-history_days:]

    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long_sales = sales_df[id_cols + value_cols].melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="d",
        value_name="demand",
    )

    aggregated = (
        long_sales.groupby(["state_id", "store_id", "cat_id", "dept_id", "d"], as_index=False)["demand"]
        .sum()
        .sort_values(["store_id", "dept_id", "d"])
    )

    calendar_cols = [
        "d",
        "date",
        "wm_yr_wk",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    calendar_df = calendar_df[calendar_cols]
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    calendar_df[["event_name_1", "event_type_1", "event_name_2", "event_type_2"]] = calendar_df[
        ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    ].fillna("none")

    item_map = sales_df[["item_id", "dept_id", "store_id"]].drop_duplicates()
    dept_prices = (
        sell_prices_df.merge(item_map, on=["item_id", "store_id"], how="inner")
        .groupby(["store_id", "dept_id", "wm_yr_wk"], as_index=False)["sell_price"]
        .mean()
        .rename(columns={"sell_price": "avg_sell_price"})
    )

    panel = aggregated.merge(calendar_df, on="d", how="left")
    panel = panel.merge(dept_prices, on=["store_id", "dept_id", "wm_yr_wk"], how="left")

    panel["day"] = panel["date"].dt.day
    panel["snap_state"] = np.select(
        [panel["state_id"] == "CA", panel["state_id"] == "TX", panel["state_id"] == "WI"],
        [panel["snap_CA"], panel["snap_TX"], panel["snap_WI"]],
        default=0,
    ).astype(int)

    group_keys = ["store_id", "dept_id"]
    panel = panel.sort_values(group_keys + ["date"], kind="mergesort")
    grouped = panel.groupby(group_keys, sort=False)["demand"]
    panel["lag_7"] = grouped.shift(7)
    panel["lag_28"] = grouped.shift(28)
    panel["rolling_mean_7"] = grouped.transform(
        lambda series: series.shift(1).rolling(7, min_periods=7).mean()
    )
    panel["rolling_mean_28"] = grouped.transform(
        lambda series: series.shift(1).rolling(28, min_periods=28).mean()
    )

    panel["avg_sell_price"] = panel["avg_sell_price"].fillna(panel["avg_sell_price"].median())
    panel = panel.dropna(subset=["lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28"])

    return panel[
        [
            "date",
            "state_id",
            "store_id",
            "cat_id",
            "dept_id",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "avg_sell_price",
            "wday",
            "month",
            "year",
            "day",
            "snap_state",
            "lag_7",
            "lag_28",
            "rolling_mean_7",
            "rolling_mean_28",
            "demand",
        ]
    ].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an M5-derived benchmark parquet")
    parser.add_argument("--download", action="store_true", help="Download missing Kaggle files")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-days", type=int, default=365)
    args = parser.parse_args()

    raw_files = ensure_m5_raw_files(args.raw_dir, download_missing=args.download)
    panel = build_store_dept_panel(
        calendar_df=pd.read_csv(raw_files["calendar.csv"]),
        sales_df=pd.read_csv(raw_files["sales_train_validation.csv"]),
        sell_prices_df=pd.read_csv(raw_files["sell_prices.csv"]),
        history_days=args.history_days,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    print(f"Saved M5-derived benchmark to {args.output}")
    print(f"Rows: {len(panel):,}")
    print(f"Columns: {len(panel.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())