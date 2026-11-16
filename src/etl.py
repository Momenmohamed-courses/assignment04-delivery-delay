"""ETL for daily order batches: raw -> Bronze -> Silver -> Gold, plus Gold validation."""

import logging
import os
from pathlib import Path

import pandas as pd
import shutil
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))

GOLD_COLUMNS = [
    "order_id",
    "order_date",
    "order_hour",
    "customer_region",
    "warehouse_id",
    "shipping_method",
    "distance_km",
    "items_count",
    "package_weight_kg",
    "order_value_egp",
    "promised_delivery_days",
    "warehouse_load_pct",
    "weather_risk_level",
    "carrier_on_time_rate",
    "km_per_promised_day",
    "value_per_item",
    "is_evening_order",
    "late_delivery",
]

def layer_path(layer: str, date: str) -> Path:
    return DATA_DIR / layer / f"orders_{date}.csv"

def evening_flag(hour):
        if pd.isna(hour):
            return None
        if hour >= 18:
            return 1
        return 0

def extract_to_bronze(date: str) -> Path:
    source = layer_path("raw", date)
    if not source.exists():
        raise FileNotFoundError(f"No raw file for {date}: {source}")

    target = layer_path("bronze", date)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)

    logger.info("Bronze: copied %s to %s", source.name, target)
    return target

def transform_to_silver(date: str) -> Path:
    source = layer_path("bronze", date)
    if not source.exists():
        raise FileNotFoundError(f"No Bronze file for {date}: {source}")

    df = pd.read_csv(source)
    rows_in = len(df)

     # Rule 1: one row per order_id
    df["order_id"] = df["order_id"].str.strip()
    df = df.drop_duplicates(subset="order_id", keep="first")
    # Rule 2: one official spelling per category
    shipping_names = {"standard": "Standard", "express": "Express", "same_day": "Same_Day"}
    df["shipping_method"] = df["shipping_method"].str.strip().str.lower().map(shipping_names)

    weather_names = {"low": "Low", "medium": "Medium", "high": "High"}
    df["weather_risk_level"] = df["weather_risk_level"].str.strip().str.lower().map(weather_names)

    for column in ["customer_region", "warehouse_id"]:
        df[column] = df[column].str.strip()
    
    # Rule 3: order_date as a real date
    df["order_date"] = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
    # Rule 4: invalid numbers become missing
    for column in ["distance_km", "items_count", "package_weight_kg", "order_value_egp"]:
        df[column] = df[column].where(df[column] > 0)

    df["order_hour"] = df["order_hour"].where(df["order_hour"].between(0, 23))
    df["promised_delivery_days"] = df["promised_delivery_days"].where(df["promised_delivery_days"] >= 1)
    df["warehouse_load_pct"] = df["warehouse_load_pct"].where(df["warehouse_load_pct"].between(0, 100))
    df["carrier_on_time_rate"] = df["carrier_on_time_rate"].where(df["carrier_on_time_rate"].between(0, 1))

    target = layer_path("silver", date)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)

    logger.info("Silver %s: %d rows in, %d rows out", date, rows_in, len(df))
    return target

def prepare_gold(date: str) -> Path:
    source = layer_path("silver", date)
    if not source.exists():
        raise FileNotFoundError(f"No Silver file for {date}: {source}")

    df = pd.read_csv(source)

        # New feature 1: how tight the delivery promise is
    df["km_per_promised_day"] = df["distance_km"] / df["promised_delivery_days"]

    # New feature 2: average value of one item
    df["value_per_item"] = df["order_value_egp"] / df["items_count"]

    # New feature 3: was the order placed in the evening (18:00 or later)?
    df["is_evening_order"] = df["order_hour"].apply(evening_flag)

    target = layer_path("gold", date)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)

    logger.info("Gold %s: %d rows, %d columns", date, len(df), len(df.columns))
    return target

def validate_gold(date: str) -> int:
    source = layer_path("gold", date)
    if not source.exists():
        raise FileNotFoundError(f"No Gold file for {date}: {source}")

    df = pd.read_csv(source)

    # Check 1: not empty
    if df.empty:
        raise ValueError(f"Gold validation failed for {date}: no rows")

    # Check 2: no required column missing
    missing_columns = []
    for column in GOLD_COLUMNS:
        if column not in df.columns:
            missing_columns.append(column)
    if missing_columns:
        raise ValueError(f"Gold validation failed for {date}: missing columns {missing_columns}")

    # Check 3: no duplicate order_id
    duplicate_count = df["order_id"].duplicated().sum()
    if duplicate_count > 0:
        raise ValueError(f"Gold validation failed for {date}: {duplicate_count} duplicate order_id(s)")
    # Check 4: every order_date equals the batch date
    wrong_dates = (df["order_date"] != date).sum()
    if wrong_dates > 0:
        raise ValueError(f"Gold validation failed for {date}: {wrong_dates} row(s) with a different order_date")
    # Check 5: every label is 0 or 1
    invalid_labels = (~df["late_delivery"].isin([0, 1])).sum()
    if invalid_labels > 0:
        raise ValueError(f"Gold validation failed for {date}: {invalid_labels} invalid late_delivery label(s)")

    logger.info("Gold %s passed validation: %d rows", date, len(df))
    return len(df)