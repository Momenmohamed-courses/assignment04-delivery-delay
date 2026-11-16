"""ETL for daily order batches: raw -> Bronze -> Silver -> Gold, plus Gold validation."""

import logging
import os
from pathlib import Path

import pandas as pd
import shutil
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))

def layer_path(layer: str, date: str) -> Path:
    return DATA_DIR / layer / f"orders_{date}.csv"

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