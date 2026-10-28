"""
test_pipeline.py

Sanity/unit tests for the preprocessing, baseline, Gradient Boosting
pipeline, prediction intervals, input validation, explanations, SQLite
job storage, status transitions, legacy migration, operational analytics,
and Flask web interface.

These test that the code runs correctly and produces sensible-shaped output
on the synthetic dataset — they are NOT user-testing or real-world validation
(see docs/testing-protocol.md and docs/simulated-validation-study.md).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import BaselineEstimator
from src.explanations import explain_prediction
from src.generate_synthetic_data import REPORTED_ISSUES, SERVICE_TYPES
from src.model import build_gbr_pipeline, build_quantile_pipelines
from src.preprocessing import (
    DATA_PATH,
    get_features_and_target,
    load_dataset,
    validate_prediction_inputs,
)
from src.train import evaluate_intervals


@pytest.fixture(scope="module")
def dataset():
    if not DATA_PATH.exists():
        pytest.skip(
            "Synthetic dataset not found — run "
            "`python src/generate_synthetic_data.py` first."
        )
    return load_dataset()


# ---------------------------------------------------------------------------
# Phase 1 + 2 Pipeline & Dataset Tests
# ---------------------------------------------------------------------------

def test_dataset_has_expected_columns(dataset):
    expected = {
        "service_type",
        "reported_issue",
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
    unseen["reported_issue"] = "CompletelyNovelIssue"
    # Should not raise, thanks to handle_unknown="ignore" in the encoder.
    preds = pipeline.predict(unseen)
    assert len(preds) == 1


# ---------------------------------------------------------------------------
# Phase 2: Quantile Prediction Intervals
# ---------------------------------------------------------------------------

def test_quantile_pipelines_and_interval_evaluation(dataset):
    X, y = get_features_and_target(dataset)
    quantile_models = build_quantile_pipelines()
    assert "lower" in quantile_models
    assert "upper" in quantile_models

    # Fit on subset for speed
    X_sub = X.head(100)
    y_sub = y.head(100)
    quantile_models["lower"].fit(X_sub, y_sub)
    quantile_models["upper"].fit(X_sub, y_sub)

    lower_preds = quantile_models["lower"].predict(X_sub)
    upper_preds = quantile_models["upper"].predict(X_sub)

    assert len(lower_preds) == 100
    assert len(upper_preds) == 100
    # On training data, upper quantile should generally be greater than or equal to lower
    assert (upper_preds >= lower_preds).mean() > 0.85

    # Test interval evaluation helper
    stats = evaluate_intervals(y_sub, lower_preds, upper_preds)
    assert "coverage" in stats
    assert "avg_width" in stats
    assert 0.0 <= stats["coverage"] <= 1.0
    assert stats["avg_width"] > 0


# ---------------------------------------------------------------------------
# Phase 2: Input Validation Tests
# ---------------------------------------------------------------------------

def test_input_validation_rules():
    # Valid input passes
    valid_data = {
        "service_type": "Screen Replacement",
        "reported_issue": "Cracked screen",
        "phone_brand": "Samsung",
        "parts_availability": "Available",
        "current_workload": "3",
        "technician_availability": "2",
        "is_warranty_case": "on",
    }
    parsed, errors = validate_prediction_inputs(valid_data)
    assert errors == []
    assert parsed is not None
    assert parsed["service_type"] == "Screen Replacement"
    assert parsed["reported_issue"] == "Cracked screen"
    assert parsed["is_warranty_case"] is True
    assert parsed["current_workload"] == 3
    assert parsed["technician_availability"] == 2

    # Missing service type
    bad_data = valid_data.copy()
    bad_data["service_type"] = ""
    _, errors = validate_prediction_inputs(bad_data)
    assert any("Service type is required" in e for e in errors)

    # Invalid reported issue for the chosen service type
    bad_data = valid_data.copy()
    bad_data["reported_issue"] = "Phone fell in water"  # Water diagnostic issue, not screen
    _, errors = validate_prediction_inputs(bad_data)
    assert any("Invalid reported issue" in e for e in errors)

    # Unknown phone brand
    bad_data = valid_data.copy()
    bad_data["phone_brand"] = "FakeBrandX"
    _, errors = validate_prediction_inputs(bad_data)
    assert any("Unknown phone brand" in e for e in errors)

    # Workload out of range
    bad_data = valid_data.copy()
    bad_data["current_workload"] = "99"
    _, errors = validate_prediction_inputs(bad_data)
    assert any("between 0 and 30" in e for e in errors)

    # Technicians out of range
    bad_data = valid_data.copy()
    bad_data["technician_availability"] = "-1"
    _, errors = validate_prediction_inputs(bad_data)
    assert any("between 0 and 10" in e for e in errors)


# ---------------------------------------------------------------------------
# Phase 2: Prediction Explanations Tests
# ---------------------------------------------------------------------------

def test_prediction_explanations():
    inputs = {
        "service_type": "Battery Replacement",
        "reported_issue": "Battery swollen",
        "phone_brand": "Apple",
        "is_warranty_case": True,
        "parts_availability": "Order Required",
        "current_workload": 12,
        "technician_availability": 0,
    }
    expl = explain_prediction(inputs, prediction_minutes=95.0)
    assert "factors" in expl
    assert "summary" in expl
    assert "customer_summary" in expl

    factor_names = [f["name"] for f in expl["factors"]]
    assert "Service type" in factor_names
    assert "Reported issue" in factor_names
    assert "Parts availability" in factor_names
    assert "Queue workload" in factor_names
    assert "Technician availability" in factor_names

    # Parts order required should be flagged as 'increase'
    parts_f = next(f for f in expl["factors"] if f["name"] == "Parts availability")
    assert parts_f["direction"] == "increase"

    # 0 technicians should be flagged as 'increase'
    tech_f = next(f for f in expl["factors"] if f["name"] == "Technician availability")
    assert tech_f["direction"] == "increase"


# ---------------------------------------------------------------------------
# Phase 2: SQLite Storage, Status Transitions & Migration Tests
# ---------------------------------------------------------------------------

def test_job_store_roundtrip(tmp_path, monkeypatch):
    """Feedback-loop sanity check: add a job, then record its actual
    completion time, using an isolated temp SQLite db."""
    import app.job_store as job_store

    temp_db = tmp_path / "jobs_test.db"
    monkeypatch.setattr(job_store, "DB_PATH", temp_db)
    monkeypatch.setattr(job_store, "JOB_LOG_PATH", tmp_path / "nonexistent.csv")

    job_id = job_store.add_job(
        {
            "service_type": "Battery Replacement",
            "reported_issue": "Battery drains fast",
            "phone_brand": "Apple",
            "is_warranty_case": False,
            "parts_availability": "Available",
            "current_workload": 2,
            "technician_availability": 1,
        },
        gbr_prediction=25.0,
        baseline_prediction=30.0,
        interval_lower=15.0,
        interval_upper=40.0,
        db_path=temp_db,
    )

    jobs = job_store.load_jobs(db_path=temp_db)
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == job_id
    assert jobs[0]["status"] == "open"
    assert jobs[0]["reported_issue"] == "Battery drains fast"
    assert jobs[0]["interval_lower_minutes"] == 15.0

    updated = job_store.record_actual_completion(job_id, 27.5, db_path=temp_db)
    assert updated is True

    jobs = job_store.load_jobs(db_path=temp_db)
    assert jobs[0]["status"] == "completed"
    assert float(jobs[0]["actual_completion_minutes"]) == 27.5


def test_job_store_status_transitions(tmp_path):
    """Verify validated status transitions in SQLite store."""
    import app.job_store as job_store

    temp_db = tmp_path / "status_test.db"

    job_id = job_store.add_job(
        {
            "service_type": "Screen Replacement",
            "reported_issue": "Cracked screen",
            "phone_brand": "Samsung",
            "is_warranty_case": False,
            "parts_availability": "Available",
            "current_workload": 1,
            "technician_availability": 2,
        },
        gbr_prediction=35.0,
        baseline_prediction=40.0,
        db_path=temp_db,
    )

    # 1. Valid: open -> in_progress
    ok, _ = job_store.update_job_status(job_id, "in_progress", db_path=temp_db)
    assert ok is True
    assert job_store.get_job(job_id, db_path=temp_db)["status"] == "in_progress"

    # 2. Invalid: in_progress -> open (not allowed)
    ok, err = job_store.update_job_status(job_id, "open", db_path=temp_db)
    assert ok is False
    assert "Cannot transition" in err

    # 3. Invalid: in_progress -> completed without actual_minutes
    ok, err = job_store.update_job_status(job_id, "completed", db_path=temp_db)
    assert ok is False
    assert "Actual completion time is required" in err

    # 4. Valid: in_progress -> completed with positive actual_minutes
    ok, _ = job_store.update_job_status(job_id, "completed", actual_minutes=32.0, db_path=temp_db)
    assert ok is True
    job = job_store.get_job(job_id, db_path=temp_db)
    assert job["status"] == "completed"
    assert float(job["actual_completion_minutes"]) == 32.0

    # 5. Invalid: completed -> any other status (terminal state)
    ok, _ = job_store.update_job_status(job_id, "open", db_path=temp_db)
    assert ok is False


def test_job_store_legacy_csv_migration(tmp_path):
    """Verify idempotent migration of legacy CSV records into SQLite."""
    import csv
    import app.job_store as job_store

    temp_csv = tmp_path / "legacy_log.csv"
    temp_db = tmp_path / "migrated.db"

    # Write 2 legacy records to CSV
    fieldnames = [
        "job_id", "created_at", "service_type", "phone_brand",
        "is_warranty_case", "parts_availability", "current_workload",
        "technician_availability", "gbr_prediction_minutes",
        "baseline_prediction_minutes", "actual_completion_minutes", "status"
    ]
    rows = [
        {
            "job_id": "legacy01",
            "created_at": "2026-09-15T08:00:00+00:00",
            "service_type": "Software Issue",
            "phone_brand": "Samsung",
            "is_warranty_case": "True",
            "parts_availability": "Available",
            "current_workload": "5",
            "technician_availability": "2",
            "gbr_prediction_minutes": "25.0",
            "baseline_prediction_minutes": "30.0",
            "actual_completion_minutes": "28.0",
            "status": "completed",
        },
        {
            "job_id": "legacy02",
            "created_at": "2026-09-15T09:00:00+00:00",
            "service_type": "Screen Replacement",
            "phone_brand": "Apple",
            "is_warranty_case": "False",
            "parts_availability": "Order Required",
            "current_workload": "8",
            "technician_availability": "1",
            "gbr_prediction_minutes": "95.0",
            "baseline_prediction_minutes": "110.0",
            "actual_completion_minutes": "",
            "status": "open",
        },
    ]
    with temp_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Initialize db with migration
    conn = job_store.ensure_db(db_path=temp_db, csv_path=temp_csv)
    conn.close()

    migrated_jobs = job_store.load_jobs(db_path=temp_db)
    assert len(migrated_jobs) == 2
    # Verify records are tagged as 'demo' source
    assert all(j["source"] == "demo" for j in migrated_jobs)

    # Idempotency check: run ensure_db again on the same db
    conn2 = job_store.ensure_db(db_path=temp_db, csv_path=temp_csv)
    conn2.close()
    assert len(job_store.load_jobs(db_path=temp_db)) == 2


def test_job_store_filters_and_search(tmp_path):
    """Test dashboard search and filtering logic."""
    import app.job_store as job_store

    temp_db = tmp_path / "filter_test.db"

    # Add 3 jobs with varied attributes
    j1 = job_store.add_job(
        {"service_type": "Battery Replacement", "reported_issue": "Swollen", "phone_brand": "Apple",
         "is_warranty_case": False, "parts_availability": "Available", "current_workload": 1, "technician_availability": 2},
        20.0, 25.0, db_path=temp_db
    )
    j2 = job_store.add_job(
        {"service_type": "Screen Replacement", "reported_issue": "Cracked screen", "phone_brand": "Samsung",
         "is_warranty_case": True, "parts_availability": "Available", "current_workload": 4, "technician_availability": 1},
        45.0, 50.0, db_path=temp_db
    )
    job_store.update_job_status(j2, "in_progress", db_path=temp_db)

    # Status filter
    open_jobs = job_store.load_jobs(status_filter="open", db_path=temp_db)
    assert len(open_jobs) == 1
    assert open_jobs[0]["job_id"] == j1

    in_prog_jobs = job_store.load_jobs(status_filter="in_progress", db_path=temp_db)
    assert len(in_prog_jobs) == 1
    assert in_prog_jobs[0]["job_id"] == j2

    # Service type filter
    battery_jobs = job_store.load_jobs(service_type_filter="Battery Replacement", db_path=temp_db)
    assert len(battery_jobs) == 1
    assert battery_jobs[0]["job_id"] == j1

    # Search query (matches brand or issue)
    samsung_jobs = job_store.load_jobs(search_query="Samsung", db_path=temp_db)
    assert len(samsung_jobs) == 1
    assert samsung_jobs[0]["job_id"] == j2


def test_job_store_operational_analytics(tmp_path):
    """Test operational analytics rules: live completed records only, >= 3 threshold."""
    import app.job_store as job_store

    temp_db = tmp_path / "analytics_test.db"

    # Case 1: 0 records -> None
    stats = job_store.get_prediction_accuracy_stats(db_path=temp_db)
    assert stats is None

    # Case 2: 2 records (< 3 threshold) -> None (prevents misleading small-sample claims)
    for i in range(2):
        jid = job_store.add_job(
            {"service_type": "Battery Replacement", "reported_issue": "Fast drain", "phone_brand": "Apple",
             "is_warranty_case": False, "parts_availability": "Available", "current_workload": 2, "technician_availability": 1},
            25.0, 30.0, db_path=temp_db
        )
        job_store.update_job_status(jid, "completed", actual_minutes=27.0, db_path=temp_db)

    stats = job_store.get_prediction_accuracy_stats(db_path=temp_db)
    assert stats is None

    # Case 3: 3rd live completed job added -> analytics now compute
    jid3 = job_store.add_job(
        {"service_type": "Battery Replacement", "reported_issue": "Fast drain", "phone_brand": "Apple",
         "is_warranty_case": False, "parts_availability": "Available", "current_workload": 2, "technician_availability": 1},
        20.0, 30.0, db_path=temp_db
    )
    job_store.update_job_status(jid3, "completed", actual_minutes=22.0, db_path=temp_db)

    stats = job_store.get_prediction_accuracy_stats(db_path=temp_db)
    assert stats is not None
    assert stats["n_records"] == 3
    assert stats["gbr_mae"] >= 0
    assert stats["baseline_mae"] >= 0

    # Service summary test
    summary = job_store.get_service_type_summary(db_path=temp_db)
    assert len(summary) >= 1
    assert summary[0]["service_type"] == "Battery Replacement"
    assert summary[0]["count"] == 3


# ---------------------------------------------------------------------------
# Phase 2: Flask App & Web Workflow Tests
# ---------------------------------------------------------------------------

def test_predict_redirects_to_bookmarkable_result_url(tmp_path, monkeypatch):
    """UI/UX fix: POST /predict must redirect (Post/Redirect/Get) to a real
    GET url, so refreshing or bookmarking the result page works instead of
    showing 'Method Not Allowed'."""
    import app.app as flask_app
    import app.job_store as job_store

    temp_db = tmp_path / "jobs_test.db"
    monkeypatch.setattr(job_store, "DB_PATH", temp_db)

    client = flask_app.app.test_client()
    response = client.post(
        "/predict",
        data={
            "service_type": "Screen Replacement",
            "reported_issue": "Cracked screen",
            "phone_brand": "Samsung",
            "parts_availability": "Available",
            "current_workload": "2",
            "technician_availability": "1",
        },
    )
    assert response.status_code == 302
    assert response.headers["Location"].startswith("/result/")

    # The redirected-to URL must be independently GET-able (e.g. on refresh).
    result_response = client.get(response.headers["Location"])
    assert result_response.status_code == 200
    assert b"ESTIMATED READY" in result_response.data
    assert b"WHAT AFFECTS THIS ESTIMATE" in result_response.data


def test_predict_validation_error_renders_form_with_errors(tmp_path, monkeypatch):
    """Invalid input should return 400 and render error list without crashing."""
    import app.app as flask_app
    import app.job_store as job_store

    temp_db = tmp_path / "jobs_test.db"
    monkeypatch.setattr(job_store, "DB_PATH", temp_db)

    client = flask_app.app.test_client()
    # Missing reported_issue
    response = client.post(
        "/predict",
        data={
            "service_type": "Screen Replacement",
            "phone_brand": "Samsung",
            "parts_availability": "Available",
            "current_workload": "2",
            "technician_availability": "1",
        },
    )
    assert response.status_code == 400
    assert b"Please fix the following" in response.data
    assert b"Reported issue is required" in response.data


def test_health_and_issues_api_endpoints():
    """Verify health monitoring and dynamic issues API."""
    import app.app as flask_app

    client = flask_app.app.test_client()

    health_res = client.get("/health")
    assert health_res.status_code == 200
    data = health_res.get_json()
    assert data["status"] == "ok"
    assert "gradient_boosting_regression" in data["model"]

    issues_res = client.get("/api/issues/Screen%20Replacement")
    assert issues_res.status_code == 200
    issues = issues_res.get_json()
    assert isinstance(issues, list)
    assert "Cracked screen" in issues


def test_full_acceptance_workflow(tmp_path, monkeypatch):
    """End-to-End Acceptance Test:
    Customer submits job -> model produces estimate & interval -> job stored in SQLite ->
    employee filters job -> transitions to in_progress -> completes with actual duration ->
    analytics calculate prediction error.
    """
    import app.app as flask_app
    import app.job_store as job_store

    temp_db = tmp_path / "workflow_test.db"
    monkeypatch.setattr(job_store, "DB_PATH", temp_db)

    client = flask_app.app.test_client()

    # Step 1: Customer submits intake
    post_res = client.post(
        "/predict",
        data={
            "service_type": "Battery Replacement",
            "reported_issue": "Battery swollen",
            "phone_brand": "Apple",
            "parts_availability": "Available",
            "current_workload": "3",
            "technician_availability": "2",
            "is_warranty_case": "on",
        },
    )
    assert post_res.status_code == 302
    result_url = post_res.headers["Location"]
    job_id = result_url.split("/")[-1]

    # Step 2: Customer views estimate and interval
    get_res = client.get(result_url)
    assert get_res.status_code == 200
    assert b"ESTIMATED READY" in get_res.data
    assert b"Typically between" in get_res.data
    assert b"Battery Replacement" in get_res.data

    # Step 3: Job stored in SQLite in 'open' state
    stored_job = job_store.get_job(job_id, db_path=temp_db)
    assert stored_job is not None
    assert stored_job["status"] == "open"
    assert stored_job["source"] == "live"
    assert stored_job["interval_lower_minutes"] is not None

    # Step 4: Employee views dashboard and filters
    dash_res = client.get("/dashboard?status=open")
    assert dash_res.status_code == 200
    assert job_id.encode() in dash_res.data

    # Step 5: Employee transitions job to in_progress
    upd_res = client.post(f"/dashboard/update/{job_id}", data={"new_status": "in_progress"})
    assert upd_res.status_code == 302
    assert job_store.get_job(job_id, db_path=temp_db)["status"] == "in_progress"

    # Step 6: Employee completes job with recorded actual duration
    comp_res = client.post(
        f"/dashboard/update/{job_id}",
        data={"new_status": "completed", "actual_completion_minutes": "26.5"},
    )
    assert comp_res.status_code == 302
    completed_job = job_store.get_job(job_id, db_path=temp_db)
    assert completed_job["status"] == "completed"
    assert float(completed_job["actual_completion_minutes"]) == 26.5


# ---------------------------------------------------------------------------
# Phase 3: Timeline Events, Tracking, Workload Intelligence & Security Tests
# ---------------------------------------------------------------------------

def test_job_events_logging_and_timeline(tmp_path):
    """Verify service timeline events are logged accurately upon each state change."""
    import app.job_store as job_store

    temp_db = tmp_path / "events_test.db"

    # 1. Job created -> logs 'created' event
    job_id = job_store.add_job(
        {
            "service_type": "Charging Port Repair",
            "reported_issue": "Loose charging port",
            "phone_brand": "OnePlus",
            "is_warranty_case": False,
            "parts_availability": "Available",
            "current_workload": 2,
            "technician_availability": 1,
        },
        gbr_prediction=28.0,
        baseline_prediction=30.0,
        interval_lower=20.0,
        interval_upper=40.0,
        db_path=temp_db,
    )

    events = job_store.get_job_events(job_id, db_path=temp_db)
    assert len(events) == 1
    assert events[0]["event_type"] == "created"
    assert "Charging Port Repair" in events[0]["details"]
    assert "28.0" in events[0]["details"]

    # 2. Transition to in_progress -> logs 'in_progress' event
    job_store.update_job_status(job_id, "in_progress", db_path=temp_db)
    events = job_store.get_job_events(job_id, db_path=temp_db)
    assert len(events) == 2
    assert events[1]["event_type"] == "in_progress"

    # 3. Transition to completed -> logs 'completed' event with actual minutes
    job_store.update_job_status(job_id, "completed", actual_minutes=25.0, db_path=temp_db)
    events = job_store.get_job_events(job_id, db_path=temp_db)
    assert len(events) == 3
    assert events[2]["event_type"] == "completed"
    assert "25.0" in events[2]["details"]


def test_workload_intelligence_and_delay_detection(tmp_path):
    """Verify queue metrics, delay tracking, and backlog calculations."""
    from datetime import datetime, timedelta, timezone
    import app.job_store as job_store

    temp_db = tmp_path / "intel_test.db"

    # Add job that was created 120 minutes ago with 30 min predicted duration
    old_time = (datetime.now(timezone.utc) - timedelta(minutes=120)).isoformat(timespec="seconds")

    conn = job_store.ensure_db(temp_db)
    conn.execute(
        """INSERT INTO jobs (job_id, created_at, service_type, reported_issue,
           phone_brand, is_warranty_case, parts_availability, current_workload,
           technician_availability, gbr_prediction_minutes,
           baseline_prediction_minutes, interval_lower_minutes,
           interval_upper_minutes, status, source)
           VALUES ('delayed1', ?, 'Screen Replacement', 'Cracked screen', 'Samsung', 0,
                   'Available', 1, 1, 30.0, 35.0, 20.0, 45.0, 'in_progress', 'live')""",
        (old_time,),
    )
    conn.commit()
    conn.close()

    delayed_job = job_store.get_job("delayed1", db_path=temp_db)
    delay_check = job_store.check_job_delay(delayed_job)
    assert delay_check["is_delayed"] is True
    assert delay_check["delay_minutes"] >= 80.0

    # Add a normal on-time job with high uncertainty (Order Required)
    job_store.add_job(
        {
            "service_type": "Battery Replacement",
            "reported_issue": "Swollen",
            "phone_brand": "Apple",
            "is_warranty_case": False,
            "parts_availability": "Order Required",
            "current_workload": 1,
            "technician_availability": 1,
        },
        gbr_prediction=90.0,
        baseline_prediction=100.0,
        interval_lower=60.0,
        interval_upper=150.0,
        db_path=temp_db,
    )

    intel = job_store.get_workload_intelligence(db_path=temp_db)
    assert intel["active_count"] == 2
    assert intel["in_progress_count"] == 1
    assert intel["open_count"] == 1
    assert intel["overdue_count"] == 1
    assert intel["high_uncertainty_count"] == 1
    assert intel["total_backlog_minutes"] == 120.0  # 30.0 + 90.0


def test_customer_ticket_tracking_workflow(tmp_path, monkeypatch):
    """Verify customer ticket lookup and live status tracking page."""
    import app.app as flask_app
    import app.job_store as job_store

    temp_db = tmp_path / "track_test.db"
    monkeypatch.setattr(job_store, "DB_PATH", temp_db)

    client = flask_app.app.test_client()

    # 1. Tracking search page
    track_home_res = client.get("/track")
    assert track_home_res.status_code == 200
    assert b"Track your repair" in track_home_res.data

    # 2. Query param search redirect
    redirect_res = client.get("/track?ticket_code=abc12345")
    assert redirect_res.status_code == 302
    assert redirect_res.headers["Location"].endswith("/track/abc12345")

    # 3. Nonexistent ticket -> 404 with helpful guidance
    not_found_res = client.get("/track/nonexistent99")
    assert not_found_res.status_code == 404
    assert b"No repair ticket found" in not_found_res.data

    # 4. Create real job and track it
    job_id = job_store.add_job(
        {
            "service_type": "Water Damage Diagnostic",
            "reported_issue": "Phone fell in water",
            "phone_brand": "Xiaomi",
            "is_warranty_case": True,
            "parts_availability": "Available",
            "current_workload": 3,
            "technician_availability": 1,
        },
        gbr_prediction=45.0,
        baseline_prediction=50.0,
        interval_lower=30.0,
        interval_upper=65.0,
        db_path=temp_db,
    )

    status_res = client.get(f"/track/{job_id}")
    assert status_res.status_code == 200
    assert b"REPAIR STATUS TRACKER" in status_res.data
    assert b"QUEUED" in status_res.data
    assert b"Water Damage Diagnostic" in status_res.data
    assert b"CHRONOLOGICAL SERVICE TIMELINE" in status_res.data
    assert job_id.encode() in status_res.data


def test_csv_export_endpoint(tmp_path, monkeypatch):
    """Verify RFC-4180 CSV export download."""
    import app.app as flask_app
    import app.job_store as job_store

    temp_db = tmp_path / "export_test.db"
    monkeypatch.setattr(job_store, "DB_PATH", temp_db)
    monkeypatch.setattr(job_store, "JOB_LOG_PATH", tmp_path / "nonexistent.csv")
    monkeypatch.setattr(job_store, "LEGACY_CSV_PATH", tmp_path / "nonexistent.csv")

    # Add 2 sample jobs
    job_store.add_job(
        {"service_type": "Screen Replacement", "reported_issue": "Cracked screen", "phone_brand": "Samsung",
         "is_warranty_case": False, "parts_availability": "Available", "current_workload": 1, "technician_availability": 2},
        35.0, 40.0, db_path=temp_db
    )
    job_store.add_job(
        {"service_type": "Battery Replacement", "reported_issue": "Fast drain", "phone_brand": "Apple",
         "is_warranty_case": True, "parts_availability": "Available", "current_workload": 2, "technician_availability": 1},
        20.0, 25.0, db_path=temp_db
    )

    client = flask_app.app.test_client()
    export_res = client.get("/dashboard/export")
    assert export_res.status_code == 200
    assert export_res.mimetype == "text/csv"
    assert "attachment; filename=repair_jobs_export.csv" in export_res.headers["Content-Disposition"]
    
    csv_text = export_res.data.decode("utf-8")
    lines = [line.strip() for line in csv_text.strip().split("\n") if line.strip()]
    assert len(lines) == 3  # header + 2 jobs
    assert "job_id,created_at,status,service_type" in lines[0]
    assert "Screen Replacement" in csv_text
    assert "Battery Replacement" in csv_text


def test_staff_authentication_and_access_control(tmp_path, monkeypatch):
    """Verify staff PIN authentication, session persistence, and logout."""
    import app.app as flask_app
    import app.job_store as job_store

    temp_db = tmp_path / "auth_test.db"
    monkeypatch.setattr(job_store, "DB_PATH", temp_db)

    client = flask_app.app.test_client()

    # 1. Login page displays
    login_get = client.get("/dashboard/login")
    assert login_get.status_code == 200
    assert b"Staff Authentication" in login_get.data

    # 2. Invalid PIN rejected
    bad_login = client.post("/dashboard/login", data={"pin": "wrong_pin"})
    assert bad_login.status_code == 200
    assert b"Invalid Staff PIN" in bad_login.data

    # 3. Correct PIN authenticates and sets session
    good_login = client.post("/dashboard/login", data={"pin": "1234"}, follow_redirects=False)
    assert good_login.status_code == 302
    assert good_login.headers["Location"].endswith("/dashboard")

    # With session set, accessing dashboard succeeds
    dash_res = client.get("/dashboard")
    assert dash_res.status_code == 200
    assert b"Staff dashboard" in dash_res.data

    # 4. Logout clears session
    logout_res = client.post("/dashboard/logout")
    assert logout_res.status_code == 302

    # 5. Access control enforcement when enabled
    flask_app.app.config["TEST_STAFF_AUTH"] = True
    try:
        protected_res = client.get("/dashboard")
        assert protected_res.status_code == 302
        assert "/dashboard/login" in protected_res.headers["Location"]
    finally:
        flask_app.app.config["TEST_STAFF_AUTH"] = False


def test_custom_404_error_page():
    """Verify custom branded 404 departure board error page."""
    import app.app as flask_app

    client = flask_app.app.test_client()
    res = client.get("/non-existent-route-xyz")
    assert res.status_code == 404
    assert b"404" in res.data
    assert b"Page or Record Not Found" in res.data

