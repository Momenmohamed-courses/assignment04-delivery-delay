"""ML training: load validated Gold data, split by date, train, evaluate and save the model."""

import logging
import json
import joblib
import pandas as pd
import shutil
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.etl import DATA_DIR, PROJECT_ROOT, validate_gold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)
ML_DIR = DATA_DIR / "ml"
MODEL_PATH = PROJECT_ROOT / "models" / "delivery_delay_model.joblib"
METRICS_PATH = PROJECT_ROOT / "metrics" / "metrics.json"
TEST_START_DATE = "2026-11-07"
TARGET = "late_delivery"
NUMERIC_FEATURES = [
    "order_hour",
    "distance_km",
    "items_count",
    "package_weight_kg",
    "order_value_egp",
    "promised_delivery_days",
    "warehouse_load_pct",
    "carrier_on_time_rate",
    "km_per_promised_day",
    "value_per_item",
    "is_evening_order",
]
CATEGORICAL_FEATURES = ["customer_region", "warehouse_id", "shipping_method", "weather_risk_level"]

def load_gold_data():
    gold_files = sorted((DATA_DIR / "gold").glob("orders_*.csv"))
    if not gold_files:
        raise FileNotFoundError(f"No Gold files in {DATA_DIR / 'gold'}; run data_pipeline first")

    tables = []
    for file in gold_files:
        date = file.stem.replace("orders_", "")
        validate_gold(date)
        tables.append(pd.read_csv(file))

    all_orders = pd.concat(tables, ignore_index=True)
    ML_DIR.mkdir(parents=True, exist_ok=True)
    target = ML_DIR / "gold_all.csv"
    all_orders.to_csv(target, index=False)

    logger.info("Loaded %d validated Gold files: %d rows", len(gold_files), len(all_orders))
    return target

def train_model():
    # 1. read ALL orders (10,053 rows, all days)
    all_orders = pd.read_csv(ML_DIR / "gold_all.csv")

    # 2. split the ROWS by date
    train_orders = all_orders[all_orders["order_date"] < TEST_START_DATE]   # Oct 1 – Nov 6
    test_orders = all_orders[all_orders["order_date"] >= TEST_START_DATE]   # Nov 7 – 15
    if train_orders.empty or test_orders.empty:
        raise ValueError(f"The split at {TEST_START_DATE} left the train or test part empty")

    # 3. build the EMPTY machine (no data yet)
    numeric_steps = Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]) 
    preparation = ColumnTransformer([
        ("numbers", numeric_steps , NUMERIC_FEATURES),
        ("words", categorical_steps, CATEGORICAL_FEATURES),
    ])
    machine = Pipeline([
        ("preparation", preparation),
        ("algorithm", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])

    # 4. pick the COLUMNS: inputs and answers, from the training rows
    input_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    train_inputs = train_orders[input_columns]       # what the machine looks at
    train_answers = train_orders[TARGET]             # what it must learn to predict

    # 5. the data goes INTO the machine here: preparation, then the algorithm learns
    machine.fit(train_inputs, train_answers)

    # 6. save the results for the next tasks
    train_orders.to_csv(ML_DIR / "train.csv", index=False)
    test_orders.to_csv(ML_DIR / "test.csv", index=False)
    joblib.dump(machine, ML_DIR / "model_candidate.joblib")

    logger.info(
        "Trained on %d rows (before %s); %d test rows kept",
        len(train_orders), TEST_START_DATE, len(test_orders),
    )

def evaluate_model():
    # 1. load the trained machine
    machine = joblib.load(ML_DIR / "model_candidate.joblib")

    # 2. load the TEST orders: inputs and real answers
    test_orders = pd.read_csv(ML_DIR / "test.csv")
    test_inputs = test_orders[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    test_answers = test_orders[TARGET]

    # 3. the machine predicts late / on time for each test order
    predictions = machine.predict(test_inputs)

    # 4. compare the guesses with the real answers (class 1 = late)
    metrics = {
        "accuracy": round(accuracy_score(test_answers, predictions), 3),
        "precision": round(precision_score(test_answers, predictions), 3),
        "recall": round(recall_score(test_answers, predictions), 3),
        "f1": round(f1_score(test_answers, predictions), 3),
    }

    # 5.  split details 
    train_orders = pd.read_csv(ML_DIR / "train.csv")
    metrics["positive_class"] = "late_delivery = 1 (late)"
    metrics["split"] = {
        "strategy": "time-based",
        "test_start_date": TEST_START_DATE,
        "train_rows": len(train_orders),
        "test_rows": len(test_orders),
        "train_dates": f"{train_orders['order_date'].min()} to {train_orders['order_date'].max()}",
        "test_dates": f"{test_orders['order_date'].min()} to {test_orders['order_date'].max()}",
    }

    # 6. save the scores for the next task
    (ML_DIR / "metrics_candidate.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Evaluation on %d test rows: %s", len(test_orders), metrics)

def save_model():
    candidate_model = ML_DIR / "model_candidate.joblib"
    candidate_metrics = ML_DIR / "metrics_candidate.json"
    if not candidate_model.exists() or not candidate_metrics.exists():
        raise FileNotFoundError("No evaluated candidate model; run train_model and evaluate_model first")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate_model, MODEL_PATH)
    shutil.copyfile(candidate_metrics, METRICS_PATH)
    logger.info("Saved model to %s and metrics to %s", MODEL_PATH, METRICS_PATH)