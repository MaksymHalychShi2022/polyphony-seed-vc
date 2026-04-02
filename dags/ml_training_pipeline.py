"""
Airflow DAG: ML Training Pipeline for seed-vc.

Pipeline:
    check_data
        → prepare_data
        → train_model
        → evaluate_model
        → branch_on_quality
            → register_model   (train/loss < CFM_LOSS_THRESHOLD)
            → stop_pipeline    (train/loss >= CFM_LOSS_THRESHOLD)

Environment variables (passed via docker-compose.airflow.yml or Airflow Variables):
    PROJECT_ROOT        — absolute path to the mounted project directory
    CFM_LOSS_THRESHOLD  — quality gate: register only when train/loss is below this value
    MLFLOW_TRACKING_URI — MLflow server URL (inherited from .env)
"""

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/airflow/project")
PYTHON = f"{PROJECT_ROOT}/.venv/bin/python"
DVC = f"{PROJECT_ROOT}/.venv/bin/dvc"

# Quality gate threshold (lower is better for CFM loss)
CFM_LOSS_THRESHOLD = float(os.environ.get("CFM_LOSS_THRESHOLD", "10.0"))

# ── DAG definition ───────────────────────────────────────────────────────────

default_args = {
    "owner": "airflow",
    "retries": 1,
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="seed-vc: data check → feature prep → training → eval → registration",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Triggered manually or via CI webhook
    catchup=False,
    tags=["seed-vc", "training", "mlops"],
) as dag:
    # ──────────────────────────────────────────────────────────────────────────
    # Task 1 — Sensor / Check: verify training data exists
    # ──────────────────────────────────────────────────────────────────────────
    def _check_data() -> None:
        train_csv = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
        if not os.path.exists(train_csv):
            raise FileNotFoundError(
                f"Training CSV not found: {train_csv}\n"
                "Run 'dvc repro process-raw' to generate it."
            )

    check_data = PythonOperator(
        task_id="check_data",
        python_callable=_check_data,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Task 2 — Data Preparation: reproduce feature extraction via DVC
    # ──────────────────────────────────────────────────────────────────────────
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{DVC} repro extract-mel extract-semantic extract-f0 extract-embedding"
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Task 3 — Model Training
    # ──────────────────────────────────────────────────────────────────────────
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(f"cd {PROJECT_ROOT} && {PYTHON} -m seed_vc.train.train"),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Task 4 — Evaluation: read latest MLflow run metrics → XCom
    # ──────────────────────────────────────────────────────────────────────────
    def _evaluate_model(**context) -> dict:
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        experiments = client.search_experiments()
        if not experiments:
            raise RuntimeError("No MLflow experiments found.")

        all_runs = []
        for exp in experiments:
            all_runs.extend(
                client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1,
                )
            )

        if not all_runs:
            raise RuntimeError("No MLflow runs found after training.")

        latest_run = max(all_runs, key=lambda r: r.info.start_time)
        metrics = {k: float(v) for k, v in latest_run.data.metrics.items()}

        ti = context["ti"]
        ti.xcom_push(key="run_id", value=latest_run.info.run_id)
        ti.xcom_push(key="metrics", value=metrics)
        return metrics

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Task 5 — Branching: route based on quality gate
    # ──────────────────────────────────────────────────────────────────────────
    def _branch_on_quality(**context) -> str:
        metrics = context["ti"].xcom_pull(task_ids="evaluate_model", key="metrics")
        loss = float(metrics.get("train/loss", float("inf")))
        if loss < CFM_LOSS_THRESHOLD:
            return "register_model"
        return "stop_pipeline"

    branch_on_quality = BranchPythonOperator(
        task_id="branch_on_quality",
        python_callable=_branch_on_quality,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Task 6a — Model Registration: register in MLflow Model Registry (Staging)
    # ──────────────────────────────────────────────────────────────────────────
    def _register_model(**context) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        run_id = context["ti"].xcom_pull(task_ids="evaluate_model", key="run_id")
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        model_uri = f"runs:/{run_id}/model"
        registered = mlflow.register_model(model_uri=model_uri, name="seed-vc")

        client = MlflowClient()
        client.transition_model_version_stage(
            name="seed-vc",
            version=registered.version,
            stage="Staging",
        )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Task 6b — Stop Pipeline: quality gate failed, no registration
    # ──────────────────────────────────────────────────────────────────────────
    stop_pipeline = EmptyOperator(task_id="stop_pipeline")

    # ──────────────────────────────────────────────────────────────────────────
    # Wire task dependencies
    # ──────────────────────────────────────────────────────────────────────────
    (
        check_data
        >> prepare_data
        >> train_model
        >> evaluate_model
        >> branch_on_quality
        >> [register_model, stop_pipeline]
    )
