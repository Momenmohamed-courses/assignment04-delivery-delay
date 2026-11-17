# Delivery Delay Prediction: From Daily CSVs to Automated ML Pipelines

Assignment 04 of the Production AI Engineering course. The goal: a **reliable, date-aware pipeline** that turns daily order files into ML-ready data and a trained model, orchestrated with **Apache Airflow** in **Docker**. The focus is reproducible engineering, not the highest accuracy.

## The scenario

A delivery company receives one CSV of orders per day. It wants to predict whether an order will be **delivered late** (`late_delivery`: 0 = on time, 1 = late), using only what is known when the order is placed.

## Dataset

- 46 daily files, `data/raw/orders_2026-10-01.csv` to `orders_2026-11-15.csv`, same 15-column schema (described in `data_dictionary.csv`)
- 10,053 orders after cleaning
- The raw files are never modified: every pipeline step reads them and writes elsewhere

## Stage A: data assessment

I inspected three batches (2026-10-01, 2026-10-18, 2026-11-15; 683 rows) in `notebooks/01_data_assessment.ipynb`.

**Column roles**
- Identifier: `order_id` · Date: `order_date` (must equal the batch date) · Target: `late_delivery`
- Numeric features: order_hour, distance_km, items_count, package_weight_kg, order_value_egp, promised_delivery_days, warehouse_load_pct, carrier_on_time_rate
- Categorical features: customer_region, warehouse_id, shipping_method, weather_risk_level

**What I found**
1. **Missing values** (all ≤ 2%): distance_km 14, warehouse_load_pct 13, carrier_on_time_rate 10, weather_risk_level 9, shipping_method 3. ID, date and label are never missing.
2. **Invalid values:** 6 negative distances, 5 orders with 0 items, 1 negative order value.
3. **Duplicates:** 9 order_ids appear twice, each an exact copy.
4. **Category spellings:** shipping_method has variants (" standard", "EXPRESS", "same_day").
5. **Dates:** every order_date matched its batch date.
6. **Label balance:** 73.6% on time, 26.4% late, so accuracy alone is misleading.

**Where each problem is handled**
- **Silver** fixes things that are simply wrong, with fixed rules (nothing learned from the data).
- **ML training** handles anything *learned* from data (filling gaps with a median, scaling, encoding), fitted on training rows only, to avoid leakage.

## Medallion layers (`src/etl.py`)

One file per day in every layer: `data/<layer>/orders_<YYYY-MM-DD>.csv`. Each run **overwrites** its day's file, so re-running a date never duplicates rows.

**🥉 Bronze:** an exact byte-for-byte copy of the raw file, bad records and duplicates included. It's the safety net to re-run from.

**🥈 Silver rules**
1. One row per `order_id` (IDs trimmed, first row kept)
2. One spelling per category (trim, lowercase, map to `Standard` / `Express` / `Same_Day` and `Low` / `Medium` / `High`; unknown values become missing)
3. `order_date` converted to a real date (unreadable dates become missing)
4. Impossible numbers become missing: distance, items, weight and value must be > 0; hour 0–23; promised days ≥ 1; load 0–100%; on-time rate 0–1
5. Missing values stay missing (filled later, inside training)

**🥇 Gold:** Silver plus three engineered features, all computed from the order's own values (no leakage):
- `km_per_promised_day` = distance / promised days: how tight the promise is
- `value_per_item` = order value / items: expensive single items vs many cheap ones
- `is_evening_order` = 1 if placed at 18:00 or later (a missing hour stays missing, it's not guessed as 0)

**Validation** (`validate_gold`) fails clearly on: no rows, a missing required column, duplicate order_ids, an order_date different from the batch date, or a label that isn't 0/1. Each check was tested against a deliberately broken file.

## The two Airflow DAGs

**`data_pipeline`** (`dags/data_pipeline.py`)

```
extract_to_bronze → transform_to_silver → prepare_gold → validate_gold
```

- daily schedule, `start_date` 2026-10-01, `end_date` 2026-11-15, `catchup=False` (past dates are processed on purpose, by trigger or backfill)
- every task gets the **run's logical date** and processes only that day's file
- `retries=2` with a 1-minute delay; `max_active_runs=1`
- an optional `simulate_failure` parameter makes `extract_to_bronze` fail on its first attempt only, to demonstrate retries safely

**`ml_training_pipeline`** (`dags/ml_training_pipeline.py`)

```
load_gold_data → train_model → evaluate_model → save_model
```

- triggered manually once validated Gold data exists
- tasks hand over work through files in `data/ml/` (intermediate files, recreated by every run)

## Model (`src/train.py`)

- **Data:** all 46 Gold files, each re-validated before training
- **Split: time-based.** Train on 2026-10-01 to 11-06 (7,890 rows), test on 11-07 to 11-15 (2,163 rows). This mimics real use (learn from the past, predict the future) and avoids same-day information leaking between train and test.
- **Preprocessing** (one scikit-learn Pipeline, fitted on training rows only):
  - numbers: fill gaps with the median, then scale
  - categories: fill gaps with the most frequent value, then one-hot encode
- **Classifier:** Logistic Regression with `class_weight="balanced"`, so the rarer late orders aren't ignored. Simple and explainable.
- `order_id` and `order_date` are not used as features.
- A model only becomes official (`save_model`) after it has been evaluated.

## Results

From `metrics/metrics.json` (class 1 = late):

| Metric | Value |
|---|---|
| Accuracy | 0.577 |
| Precision | 0.385 |
| Recall | 0.772 |
| F1 | 0.514 |

**How I read them:** the model catches **77% of late deliveries**, at the cost of many false alarms (precision 0.39). Accuracy is lower than a model that always says "on time" (71% in the test period), but that model would catch zero late orders. Whether this trade-off is right depends on the business: it's useful if a missed delay is costly and an extra alert is cheap.

Also worth knowing: late orders are more frequent in the test period (28.9%) than in training (22.2%), so the test is a slightly harder, more realistic period.

## Evidence (`screenshots/`)

| File | Shows |
|---|---|
| `01_data_pipeline_graph_success.png` | data_pipeline graph, all tasks successful |
| `02_idempotency_before.png` / `04_idempotency_after.png` | row counts and duplicate checks before and after re-running 2026-10-01: identical counts, no duplicate IDs in Silver/Gold, files rewritten, raw file untouched |
| `03_idempotency_rerun_ui.png` | the re-run in Airflow |
| `05_backfill_grid.png` | backfill of 2026-10-18, 19 and 20 |
| `06_backfill_gold_partitions.png` | the three resulting Gold partitions |
| `07_retry_attempt1_failed.png` / `08_retry_attempt2_success.png` | simulated temporary failure on attempt 1, success on retry |
| `09_ml_pipeline_graph_success.png` | ml_training_pipeline graph, all tasks successful |
| `10_ml_evaluate_log.png` | evaluation log with the metrics |

All 46 dates were processed through Airflow, and all passed Gold validation.

## Project structure

```
assignment04-delivery-delay/
├── dags/
│   ├── data_pipeline.py          # daily Bronze → Silver → Gold → validation
│   └── ml_training_pipeline.py   # load → train → evaluate → save
├── src/
│   ├── etl.py                    # dated Bronze/Silver/Gold/validation functions
│   └── train.py                  # load, split, train, evaluate, save
├── data/
│   ├── raw/                      # the 46 original CSVs (never modified)
│   ├── bronze/ silver/ gold/     # one file per date, generated by data_pipeline
│   └── ml/                       # intermediate files of ml_training_pipeline
├── models/delivery_delay_model.joblib
├── metrics/metrics.json
├── notebooks/01_data_assessment.ipynb
├── screenshots/
├── Dockerfile                    # Airflow 3.4.0 + pinned project libraries
├── docker-compose.yml            # Postgres + Airflow (LocalExecutor)
├── requirements.txt
├── .env.example
├── data_dictionary.csv
└── README.md
```

## How to reproduce

### Prerequisites
- Docker Desktop with at least 4 GB of memory for Docker (8 GB recommended)
- Git, and Python 3 for generating the keys below

### 1. Clone and configure

```bash
git clone <this-repo-url>
cd assignment04-delivery-delay
cp .env.example .env
```

Fill in `.env` (instructions are in the file):
- `AIRFLOW_UID` → the output of `id -u`
- `FERNET_KEY` → `python3 -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"`
- `AIRFLOW__API_AUTH__JWT_SECRET` → `python3 -c "import secrets; print(secrets.token_hex(32))"`

### 2. Start Airflow

```bash
docker compose up -d --build
docker compose ps
```

Wait until all services show `healthy`, then open **http://localhost:8081** and log in with `airflow` / `airflow`.

(The web UI is mapped to port **8081** on the host; if that port is busy on your machine, change the left number of `"8081:8080"` in `docker-compose.yml`.)

### 3. Run the data pipeline

In the UI, switch `data_pipeline` on. To process all 46 days:

```bash
docker compose exec airflow-scheduler airflow backfill create --dag-id data_pipeline --from-date 2026-10-01 --to-date 2026-11-15T12:00:00
```

To process a single date, use **Trigger** with a logical date at **12:00** (noon avoids the UI's local timezone shifting the date into the previous UTC day).

**Evidence runs:**
- **Same-date rerun:** open a successful run → **Clear** → it re-runs with the same date
- **Backfill of three days:**
  ```bash
  docker compose exec airflow-scheduler airflow backfill create --dag-id data_pipeline --from-date 2026-10-18 --to-date 2026-10-20T12:00:00
  ```
- **Retry:** trigger a run with the `simulate_failure` box ticked; `extract_to_bronze` fails once, then succeeds on its retry

### 4. Train the model

Switch `ml_training_pipeline` on and click **Trigger**. Results:
- `models/delivery_delay_model.joblib`
- `metrics/metrics.json`

### 5. Stop Airflow

```bash
docker compose down        # stop everything (Airflow's history is kept)
docker compose down -v     # stop and also delete Airflow's database
```

### Optional: work locally without Airflow

```bash
uv venv --python 3.13
uv pip install -r requirements.txt jupyter ipykernel
.venv/bin/python -c "from src.etl import extract_to_bronze; extract_to_bronze('2026-10-18')"
```

## Limitations and next steps

- One small dataset (46 days); a longer history would make the time-based evaluation more reliable.
- Logistic Regression is a simple baseline; a tree-based model (e.g. Random Forest) or tuning the decision threshold could balance precision and recall better.
- The ML DAG is triggered manually; a next step would be triggering it automatically after new validated Gold data arrives.