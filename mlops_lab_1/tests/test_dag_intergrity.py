from airflow.models import DagBag


def test_dag_integrity():
    dag_bag = DagBag(dag_folder="mlops_lab_1/dags", include_examples=False)
    assert dag_bag.import_errors == {}
    assert "ml_training_pipeline" in dag_bag.dags