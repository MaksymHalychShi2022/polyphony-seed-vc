"""DAG integrity tests — verifies that Airflow can load all DAG files without errors."""

import os

DAGS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "dags")


def test_dag_no_import_errors():
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder=DAGS_FOLDER, include_examples=False)
    assert len(dag_bag.import_errors) == 0, "DAG import errors:\n" + "\n".join(
        f"  {path}: {err}" for path, err in dag_bag.import_errors.items()
    )


def test_dag_loaded():
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder=DAGS_FOLDER, include_examples=False)
    assert "ml_training_pipeline" in dag_bag.dags, (
        f"DAG 'ml_training_pipeline' not found. "
        f"Loaded DAGs: {list(dag_bag.dags.keys())}"
    )


def test_dag_task_ids():
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder=DAGS_FOLDER, include_examples=False)
    dag = dag_bag.dags["ml_training_pipeline"]
    task_ids = set(dag.task_ids)
    required = {
        "check_data",
        "prepare_data",
        "train_model",
        "evaluate_model",
        "branch_on_quality",
        "register_model",
        "stop_pipeline",
    }
    missing = required - task_ids
    assert not missing, f"Missing tasks in DAG: {missing}"
