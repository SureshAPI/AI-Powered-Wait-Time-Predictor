"""
test_pipeline.py

Sanity/unit tests for the preprocessing, baseline, and Gradient Boosting
pipeline. These test that the code runs correctly and produces
sensible-shaped output on the synthetic dataset — they are NOT user-testing
or real-world validation (see docs/testing.md, added at the user-testing
milestone).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import BaselineEstimator
from src.model import build_gbr_pipeline
from src.preprocessing import (
    DATA_PATH,
    get_features_and_target,
    load_dataset,
)


@pytest.fixture(scope="module")
def dataset():
    if not DATA_PATH.exists():
        pytest.skip(
            "Synthetic dataset not found — run "
            "`python src/generate_synthetic_data.py` first."
        )
    return load_dataset()


def test_dataset_has_expected_columns(dataset):
    expected = {
        "service_type",
        "phone_brand",
        "is_warranty_case",
        "parts_availability",
        "current_workload",
        "technician_availability",
        "actual_duration_minutes",
    }
    assert expected.issubset(set(dataset.columns))


def test_dataset_target_is_positive(dataset):
    assert (dataset["actual_duration_minutes"] > 0).all()


def test_baseline_predicts_reasonable_values(dataset):
    X, y = get_features_and_target(dataset)
    baseline = BaselineEstimator().fit(X, y)
    preds = baseline.predict(X.head(10))
    assert len(preds) == 10
    assert all(p >= 5.0 for p in preds)


def test_gbr_pipeline_fits_and_predicts(dataset):
    X, y = get_features_and_target(dataset)
    pipeline = build_gbr_pipeline()
    pipeline.fit(X, y)
    preds = pipeline.predict(X.head(5))
    assert len(preds) == 5
    assert all(p > 0 for p in preds)


def test_gbr_handles_unseen_category_gracefully(dataset):
    X, y = get_features_and_target(dataset)
    pipeline = build_gbr_pipeline()
    pipeline.fit(X, y)

    unseen = X.head(1).copy()
    unseen["phone_brand"] = "TotallyNewBrand"
    # Should not raise, thanks to handle_unknown="ignore" in the encoder.
    preds = pipeline.predict(unseen)
    assert len(preds) == 1


def test_job_store_roundtrip(tmp_path, monkeypatch):
    """Feedback-loop sanity check: add a job, then record its actual
    completion time, using an isolated temp file so this test never touches
    the real data/job_log.csv."""
    import app.job_store as job_store

    temp_log = tmp_path / "job_log_test.csv"
    monkeypatch.setattr(job_store, "JOB_LOG_PATH", temp_log)

    job_id = job_store.add_job(
        {
            "service_type": "Battery Replacement",
            "phone_brand": "Apple",
            "is_warranty_case": False,
            "parts_availability": "Available",
            "current_workload": 2,
            "technician_availability": 1,
        },
        gbr_prediction=25.0,
        baseline_prediction=30.0,
    )

    jobs = job_store.load_jobs()
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == job_id
    assert jobs[0]["status"] == "open"

    updated = job_store.record_actual_completion(job_id, 27.5)
    assert updated is True

    jobs = job_store.load_jobs()
    assert jobs[0]["status"] == "completed"
    assert jobs[0]["actual_completion_minutes"] == "27.5"
