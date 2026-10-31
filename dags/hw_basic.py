from datetime import timedelta

import pendulum
import requests
from airflow.providers.postgres.hooks.postgres import PostgresHook

from batch_common import (
    business_date,
    digest,
    save,
    read,
    validate_rows,
    infer_batches,
    validate_predictions,
    replace_day,
)
from airflow.sdk import dag, task, Param, BaseHook, get_current_context


@dag(
    dag_id="hw_basic_batch_inference",
    start_date=pendulum.datetime(
        2026, 9, 1, 0, 0, tz="Europe/Moscow"
    ),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["homework", "basic"],
    params={
        "business_date": Param(
            "2026-09-01",
            type="string",
            format="date",
        ),
        "batch_size": Param(
            32,
            type="integer",
            minimum=1,
            maximum=1000,
        ),
    },
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=5),
        "execution_timeout": timedelta(minutes=2),
    },
)
def hw_basic_batch_inference():

    @task(task_id="extract")
    def extract():
        context = get_current_context()
        params = context["params"]

        day = business_date(
            params["business_date"],
            context.get("data_interval_start"),
        )

        connection = BaseHook.get_connection("source_api")
        host = connection.host.rstrip("/")

        data_response = requests.get(
            f"{host}/basic/{day}.json",
            timeout=10,
        )
        data_response.raise_for_status()
        data = data_response.json()

        if data["business_date"] != day:
            raise ValueError("Business date mismatch")

        model_response = requests.get(
            f"{host}/models/v1.json",
            timeout=10,
        )
        model_response.raise_for_status()

        model = model_response.json()
        model_sha256 = digest(model_response.content)

        return save({
            "business_date": day,
            "rows": data["rows"],
            "model": model,
            "model_sha256": model_sha256,
        })

    @task(task_id="infer", retries=0)
    def infer(extract_ref):
        context = get_current_context()
        batch_size = context["params"]["batch_size"]

        data = read(extract_ref)
        day = data["business_date"]

        rows = validate_rows(data["rows"], day)

        predictions = infer_batches(
            rows,
            data["model"],
            batch_size,
        )

        expected_ids = [row["customer_id"] for row in rows]
        validate_predictions(predictions, expected_ids)

        return save({
            "business_date": day,
            "model_version": "v1",
            "predictions": predictions,
            "model_sha256": data["model_sha256"],
        })

    @task(task_id="publish")
    def publish(infer_ref):
        result = read(infer_ref)

        conn = PostgresHook(
            postgres_conn_id="warehouse"
        ).get_conn()

        try:
            replace_day(
                conn,
                result["business_date"],
                result["model_version"],
                result["predictions"],
            )
        finally:
            conn.close()

        return {
            "business_date": result["business_date"],
            "model_version": result["model_version"],
            "rows": len(result["predictions"]),
        }

    publish(infer(extract()))


hw_basic_batch_inference()
