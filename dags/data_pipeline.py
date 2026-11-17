from datetime import datetime, timedelta

from airflow.sdk import dag, get_current_context, task

from src.etl import extract_to_bronze, prepare_gold, transform_to_silver, validate_gold

def run_date() -> str:
    logical_date = get_current_context().get("logical_date")
    if logical_date is None:
        raise ValueError("This run has no logical date; trigger it for a specific date.")
    return logical_date.strftime("%Y-%m-%d")

@dag(
    dag_id="data_pipeline",
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    end_date=datetime(2026, 11, 15),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["assignment04"],
)
def data_pipeline():
    @task(task_id="extract_to_bronze")
    def bronze():
        extract_to_bronze(run_date())

    @task(task_id="transform_to_silver")
    def silver():
        transform_to_silver(run_date())

    @task(task_id="prepare_gold")
    def gold():
        prepare_gold(run_date())

    @task(task_id="validate_gold")
    def validate():
        validate_gold(run_date())

    bronze() >> silver() >> gold() >> validate()


data_pipeline()