"""
job_store.py

Lightweight CSV-backed store for jobs created through the prototype app.
This is intentionally simple (no database) — appropriate for a college
prototype — but it is what makes the feedback loop from the original
architecture (docs/architecture.md) actually work: each prediction is logged,
and an employee can later record the real completion time against it.

Runtime data lives in data/job_log.csv, which is git-ignored (it's mutable
local state generated while using the prototype, not a tracked dataset).
data/job_log.example.csv is a small, clearly-labelled example/schema
reference that IS committed.
"""

import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

JOB_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "job_log.csv"

FIELDNAMES = [
    "job_id",
    "created_at",
    "service_type",
    "phone_brand",
    "is_warranty_case",
    "parts_availability",
    "current_workload",
    "technician_availability",
    "gbr_prediction_minutes",
    "baseline_prediction_minutes",
    "actual_completion_minutes",
    "status",
]


def _ensure_file():
    JOB_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not JOB_LOG_PATH.exists():
        with JOB_LOG_PATH.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def add_job(inputs: dict, gbr_prediction: float, baseline_prediction: float) -> str:
    """Log a new prediction. Returns the generated job_id."""
    _ensure_file()
    job_id = uuid.uuid4().hex[:8]
    row = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "service_type": inputs["service_type"],
        "phone_brand": inputs["phone_brand"],
        "is_warranty_case": inputs["is_warranty_case"],
        "parts_availability": inputs["parts_availability"],
        "current_workload": inputs["current_workload"],
        "technician_availability": inputs["technician_availability"],
        "gbr_prediction_minutes": round(gbr_prediction, 1),
        "baseline_prediction_minutes": round(baseline_prediction, 1),
        "actual_completion_minutes": "",
        "status": "open",
    }
    with JOB_LOG_PATH.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
    return job_id


def load_jobs() -> list[dict]:
    """Return all logged jobs, most recent first."""
    _ensure_file()
    with JOB_LOG_PATH.open() as f:
        jobs = list(csv.DictReader(f))
    return list(reversed(jobs))


def record_actual_completion(job_id: str, actual_minutes: float) -> bool:
    """Feedback-loop write: record the real completion time for a job.
    Returns True if the job was found and updated."""
    _ensure_file()
    with JOB_LOG_PATH.open() as f:
        jobs = list(csv.DictReader(f))

    found = False
    for job in jobs:
        if job["job_id"] == job_id:
            job["actual_completion_minutes"] = round(actual_minutes, 1)
            job["status"] = "completed"
            found = True

    if found:
        with JOB_LOG_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(jobs)

    return found
