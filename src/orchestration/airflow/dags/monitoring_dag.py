from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="stock_prediction__production_monitoring",
    default_args=default_args,
    start_date=datetime(2026, 5, 25),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "stocks"],
) as dag:

    production_monitor = BashOperator(
        task_id="production_monitor",
        bash_command="""
        cd /opt/airflow/project &&
        uv run production-monitor
        """
    )



    (
        production_monitor
    )