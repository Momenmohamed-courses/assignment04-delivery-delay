"""ML training pipeline: validated Gold data -> trained, evaluated and saved model. Triggered manually."""

from datetime import datetime, timedelta

from airflow.sdk import dag, task

from src.train import evaluate_model, load_gold_data, save_model, train_model


@dag(
    dag_id="ml_training_pipeline",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["assignment04"],
)
def ml_training_pipeline():
    @task(task_id="load_gold_data")
    def load():
        load_gold_data()

    @task(task_id="train_model")
    def train():
        train_model()

    @task(task_id="evaluate_model")
    def evaluate():
        evaluate_model()

    @task(task_id="save_model")
    def save():
        save_model()

    load() >> train() >> evaluate() >> save()


ml_training_pipeline()