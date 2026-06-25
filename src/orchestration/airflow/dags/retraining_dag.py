from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="stock_prediction_retraining",
    default_args=default_args,
    start_date=datetime(2026, 5, 25),
    schedule = "0 2 */3 * *",  # every 3 days at 2am
    catchup=False,
    tags=["mlops", "stocks"],
) as dag:

    download_data = BashOperator(
        task_id="download_data",
        bash_command="""
        cd /opt/airflow/project &&
        uv run download-update-data
        """
    )
    
    generate_signal = BashOperator(
        task_id="generate_signal",
        bash_command="""
        cd /opt/airflow/project &&
        uv run generate-signal
        """
    )

    process_data = BashOperator(
        task_id="process_data",
        bash_command="""
        cd /opt/airflow/project &&
        uv run process-data
        """
    )

    retrain_pipeline = BashOperator(
        task_id="retrain_pipeline",
        bash_command="""
        cd /opt/airflow/project &&
        uv run retrain-pipeline
        """
    )



    (
        download_data
        >> generate_signal
        >> process_data
        >> retrain_pipeline
    )