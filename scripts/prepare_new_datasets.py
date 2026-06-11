"""Preprocess the 3 new Kaggle datasets into parquet files for SCE evaluation."""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/parquet")


def prepare_rossmann():
    """Rossmann Store Sales: German retail, store-level daily sales.

    Hierarchy: StoreType → Assortment → Store → DayOfWeek
    Target: Sales (daily revenue)
    """
    train = pd.read_csv(RAW_DIR / "rossmann/train.csv", low_memory=False)
    store = pd.read_csv(RAW_DIR / "rossmann/store.csv")

    df = train.merge(store, on="Store", how="left")

    # Filter: only open stores with positive sales
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # Parse date for temporal features
    df["Date"] = pd.to_datetime(df["Date"])
    df["month"] = df["Date"].dt.month
    df["day_of_month"] = df["Date"].dt.day
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)

    # Convert DayOfWeek to string categorical
    df["DayOfWeek"] = df["DayOfWeek"].astype(str)
    df["Promo"] = df["Promo"].astype(str)
    df["Promo2"] = df["Promo2"].astype(str)
    df["SchoolHoliday"] = df["SchoolHoliday"].astype(str)
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    df["Store"] = df["Store"].astype(str)

    # Fill competition distance NaN with median
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )

    # Select final columns
    keep_cols = [
        "Date", "Store", "StoreType", "Assortment", "DayOfWeek",
        "Promo", "Promo2", "StateHoliday", "SchoolHoliday",
        "CompetitionDistance", "Customers",
        "month", "day_of_month", "week_of_year",
        "Sales",
    ]
    df = df[keep_cols].reset_index(drop=True)

    out = OUT_DIR / "rossmann_daily.parquet"
    df.to_parquet(out, index=False)
    print(f"Rossmann: {df.shape[0]:,} rows × {df.shape[1]} cols → {out}")
    return df


def prepare_walmart():
    """Walmart Sales Forecast: US retail, store-department weekly sales.

    Hierarchy: Type → Store → Dept
    Target: Weekly_Sales
    """
    train = pd.read_csv(RAW_DIR / "walmart/train.csv")
    stores = pd.read_csv(RAW_DIR / "walmart/stores.csv")
    features = pd.read_csv(RAW_DIR / "walmart/features.csv")

    # Merge all tables
    df = train.merge(stores, on="Store", how="left")
    df = df.merge(features, on=["Store", "Date", "IsHoliday"], how="left")

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"])
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["year"] = df["Date"].dt.year

    # Convert categoricals to string
    df["Store"] = df["Store"].astype(str)
    df["Dept"] = df["Dept"].astype(str)
    df["IsHoliday"] = df["IsHoliday"].astype(str)

    # Fill markdown NaNs with 0 (no markdown applied)
    for col in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
        df[col] = df[col].fillna(0.0)

    # Select final columns
    keep_cols = [
        "Date", "Store", "Dept", "Type", "IsHoliday",
        "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "month", "week_of_year", "year",
        "Weekly_Sales",
    ]
    df = df[keep_cols].reset_index(drop=True)

    out = OUT_DIR / "walmart_weekly.parquet"
    df.to_parquet(out, index=False)
    print(f"Walmart:  {df.shape[0]:,} rows × {df.shape[1]} cols → {out}")
    return df


def prepare_melbourne():
    """Melbourne Housing Market: Australian real estate prices.

    Hierarchy: Regionname → CouncilArea → Suburb → Type
    Target: Price
    """
    df = pd.read_csv(RAW_DIR / "melbourne/Melbourne_housing_FULL.csv")

    # Drop rows without price (our target)
    df = df.dropna(subset=["Price"]).copy()

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    # Fill numeric NaNs with median
    numeric_fill = ["Distance", "Bedroom2", "Bathroom", "Car", "Landsize",
                     "BuildingArea", "YearBuilt", "Propertycount"]
    for col in numeric_fill:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical NaNs
    df["CouncilArea"] = df["CouncilArea"].fillna("Unknown")
    df["Regionname"] = df["Regionname"].fillna("Unknown")

    # Convert Postcode to string categorical
    df["Postcode"] = df["Postcode"].fillna(0).astype(int).astype(str)

    # Select final columns
    keep_cols = [
        "Date", "Suburb", "Type", "Method", "Regionname", "CouncilArea",
        "Postcode",
        "Rooms", "Distance", "Bedroom2", "Bathroom", "Car",
        "Landsize", "BuildingArea", "YearBuilt", "Propertycount",
        "month", "year",
        "Price",
    ]
    df = df[keep_cols].reset_index(drop=True)

    out = OUT_DIR / "melbourne_housing.parquet"
    df.to_parquet(out, index=False)
    print(f"Melbourne: {df.shape[0]:,} rows × {df.shape[1]} cols → {out}")
    return df


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_rossmann()
    prepare_walmart()
    prepare_melbourne()
    print("\nAll datasets prepared.")
