"""
job_store.py

Phase 2: SQLite-backed persistent store for service jobs, replacing the
Phase 1 CSV-based store.

Key changes from Phase 1:
  - SQLite database instead of CSV (atomic writes, proper queries).
  - Safe, idempotent CSV → SQLite migration on first run.
  - Multiple job statuses: open → in_progress → completed (or → cancelled).
  - Validated status transitions.
  - reported_issue and prediction interval fields.
  - Search and filter support for the dashboard.
  - Analytics queries (only on real operational records).
  - Source tagging: 'live' for app-created jobs, 'demo' for migrated/test.

The database file lives at data/jobs.db (git-ignored). Legacy CSV records
in data/job_log.csv are migrated once on first access and then the CSV
is left in place (not deleted) for safety.
"""

import csv
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "jobs.db"
LEGACY_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "job_log.csv"
JOB_LOG_PATH = LEGACY_CSV_PATH  # Backward-compatible alias for Phase 1

# Valid status transitions
VALID_TRANSITIONS = {
    "open": {"in_progress", "completed", "cancelled"},
    "in_progress": {"completed", "cancelled"},
    "completed": set(),      # terminal state
    "cancelled": {"open"},   # allow reopening a cancelled job
}

ALL_STATUSES = {"open", "in_progress", "completed", "cancelled"}

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    service_type TEXT NOT NULL,
    reported_issue TEXT NOT NULL DEFAULT '',
    phone_brand TEXT NOT NULL,
    is_warranty_case INTEGER NOT NULL DEFAULT 0,
    parts_availability TEXT NOT NULL,
    current_workload INTEGER NOT NULL,
    technician_availability INTEGER NOT NULL,
    gbr_prediction_minutes REAL NOT NULL,
    baseline_prediction_minutes REAL NOT NULL,
    interval_lower_minutes REAL,
    interval_upper_minutes REAL,
    actual_completion_minutes REAL,
    status TEXT NOT NULL DEFAULT 'open',
    source TEXT NOT NULL DEFAULT 'live',
    updated_at TEXT
);
"""

_CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_service_type ON jobs(service_type);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);",
]


def get_db(db_path: Path = None) -> sqlite3.Connection:
    """Get a SQLite connection with row factory set to dict-like access."""
    if db_path is not None:
        path = db_path
    elif JOB_LOG_PATH != LEGACY_CSV_PATH:
        # Backward compatibility if a legacy test monkeypatched JOB_LOG_PATH
        path = Path(JOB_LOG_PATH).with_suffix(".db")
    else:
        path = DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn: sqlite3.Connection):
    """Create tables and indices if they don't exist."""
    conn.execute(_CREATE_TABLE_SQL)
    for idx_sql in _CREATE_INDEX_SQL:
        conn.execute(idx_sql)
    conn.commit()


def _migrate_csv_if_needed(conn: sqlite3.Connection, csv_path: Path = None):
    """One-time, idempotent migration of legacy CSV records to SQLite.

    Records migrated from CSV are tagged with source='demo' to distinguish
    them from live operational records. Migration is safe to run multiple
    times — existing job_ids are skipped.
    """
    target_csv = csv_path or (JOB_LOG_PATH if JOB_LOG_PATH != LEGACY_CSV_PATH else LEGACY_CSV_PATH)
    if not target_csv.exists():
        return

    try:
        with target_csv.open() as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return
            rows = list(reader)
    except Exception:
        return  # If CSV is malformed, skip migration silently

    if not rows:
        return

    migrated = 0
    for row in rows:
        job_id = row.get("job_id", "").strip()
        if not job_id:
            continue

        # Check if already migrated
        existing = conn.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if existing:
            continue

        # Parse fields with safe defaults
        actual = row.get("actual_completion_minutes", "").strip()
        actual_val = float(actual) if actual else None
        status = row.get("status", "open").strip()
        if status not in ALL_STATUSES:
            status = "open"

        is_warranty = row.get("is_warranty_case", "False").strip()
        is_warranty_int = 1 if is_warranty in ("True", "1", "true") else 0

        conn.execute(
            """INSERT INTO jobs (job_id, created_at, service_type, reported_issue,
               phone_brand, is_warranty_case, parts_availability, current_workload,
               technician_availability, gbr_prediction_minutes,
               baseline_prediction_minutes, actual_completion_minutes, status, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                row.get("created_at", datetime.now(timezone.utc).isoformat()),
                row.get("service_type", "Unknown"),
                row.get("reported_issue", ""),  # Phase 1 didn't have this
                row.get("phone_brand", "Unknown"),
                is_warranty_int,
                row.get("parts_availability", "Unknown"),
                int(row.get("current_workload", 0)),
                int(row.get("technician_availability", 0)),
                float(row.get("gbr_prediction_minutes", 0)),
                float(row.get("baseline_prediction_minutes", 0)),
                actual_val,
                status,
                "demo",  # tag migrated records
            ),
        )
        migrated += 1

    conn.commit()
    if migrated > 0:
        print(f"[job_store] Migrated {migrated} legacy CSV records to SQLite (tagged as 'demo').")


def ensure_db(db_path: Path = None, csv_path: Path = None) -> sqlite3.Connection:
    """Get a ready-to-use database connection: create tables, run migration."""
    conn = get_db(db_path)
    init_db(conn)
    # Only migrate default CSV when running against the default DB,
    # or when an explicit csv_path is provided.
    if csv_path is not None:
        _migrate_csv_if_needed(conn, csv_path=csv_path)
    elif db_path is None:
        _migrate_csv_if_needed(conn)
    return conn


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def add_job(inputs: dict, gbr_prediction: float, baseline_prediction: float,
            interval_lower: float = None, interval_upper: float = None,
            db_path: Path = None) -> str:
    """Log a new prediction. Returns the generated job_id."""
    conn = ensure_db(db_path)
    job_id = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    is_warranty_int = 1 if inputs.get("is_warranty_case") else 0

    conn.execute(
        """INSERT INTO jobs (job_id, created_at, service_type, reported_issue,
           phone_brand, is_warranty_case, parts_availability, current_workload,
           technician_availability, gbr_prediction_minutes,
           baseline_prediction_minutes, interval_lower_minutes,
           interval_upper_minutes, status, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', 'live')""",
        (
            job_id, now,
            inputs["service_type"],
            inputs.get("reported_issue", ""),
            inputs["phone_brand"],
            is_warranty_int,
            inputs["parts_availability"],
            int(inputs["current_workload"]),
            int(inputs["technician_availability"]),
            round(gbr_prediction, 1),
            round(baseline_prediction, 1),
            round(interval_lower, 1) if interval_lower is not None else None,
            round(interval_upper, 1) if interval_upper is not None else None,
        ),
    )
    conn.commit()
    conn.close()
    return job_id


def get_job(job_id: str, db_path: Path = None) -> dict | None:
    """Look up a single job by id, or None if not found."""
    conn = ensure_db(db_path)
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def load_jobs(status_filter: str = None, search_query: str = None,
              service_type_filter: str = None, source_filter: str = None,
              db_path: Path = None) -> list[dict]:
    """Return jobs, most recent first, with optional filters.

    Args:
        status_filter: Filter by status (open, in_progress, completed, cancelled).
        search_query: Search job_id, phone_brand, or service_type.
        service_type_filter: Filter by exact service type.
        source_filter: Filter by source ('live' or 'demo').
    """
    conn = ensure_db(db_path)
    query = "SELECT * FROM jobs WHERE 1=1"
    params = []

    if status_filter and status_filter in ALL_STATUSES:
        query += " AND status = ?"
        params.append(status_filter)

    if service_type_filter:
        query += " AND service_type = ?"
        params.append(service_type_filter)

    if source_filter and source_filter in ("live", "demo"):
        query += " AND source = ?"
        params.append(source_filter)

    if search_query:
        search_term = f"%{search_query}%"
        query += " AND (job_id LIKE ? OR phone_brand LIKE ? OR service_type LIKE ? OR reported_issue LIKE ?)"
        params.extend([search_term, search_term, search_term, search_term])

    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_job_status(job_id: str, new_status: str,
                      actual_minutes: float = None,
                      db_path: Path = None) -> tuple[bool, str]:
    """Update a job's status with validated transitions.

    Returns (success: bool, message: str).
    """
    if new_status not in ALL_STATUSES:
        return False, f"Invalid status: {new_status}"

    conn = ensure_db(db_path)
    row = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()

    if row is None:
        conn.close()
        return False, f"Job {job_id} not found."

    current_status = row["status"]
    allowed = VALID_TRANSITIONS.get(current_status, set())

    if new_status not in allowed:
        conn.close()
        return False, f"Cannot transition from '{current_status}' to '{new_status}'. Allowed: {allowed or 'none (terminal state)'}."

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if new_status == "completed":
        if actual_minutes is None:
            conn.close()
            return False, "Actual completion time is required when marking a job as completed."
        if actual_minutes <= 0:
            conn.close()
            return False, "Actual completion time must be a positive number."
        conn.execute(
            "UPDATE jobs SET status = ?, actual_completion_minutes = ?, updated_at = ? WHERE job_id = ?",
            (new_status, round(actual_minutes, 1), now, job_id),
        )
    else:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
            (new_status, now, job_id),
        )

    conn.commit()
    conn.close()
    return True, f"Job {job_id} updated to '{new_status}'."


def record_actual_completion(job_id: str, actual_minutes: float,
                             db_path: Path = None) -> bool:
    """Backward-compatible wrapper: mark a job as completed with actual time.

    Returns True if successful, False otherwise.
    """
    success, _ = update_job_status(job_id, "completed", actual_minutes, db_path)
    return success


# ---------------------------------------------------------------------------
# Analytics (Phase 2)
# ---------------------------------------------------------------------------

def get_status_counts(source_filter: str = None, db_path: Path = None) -> dict:
    """Return counts of jobs by status."""
    conn = ensure_db(db_path)
    query = "SELECT status, COUNT(*) as cnt FROM jobs"
    params = []
    if source_filter and source_filter in ("live", "demo"):
        query += " WHERE source = ?"
        params.append(source_filter)
    query += " GROUP BY status"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return {r["status"]: r["cnt"] for r in rows}


def get_prediction_accuracy_stats(db_path: Path = None) -> dict | None:
    """Compute prediction accuracy stats from jobs with real actual_completion_minutes.

    ONLY uses 'live' source records with completed status and non-null actual
    completion times. Returns None if insufficient data (< 3 records).
    """
    conn = ensure_db(db_path)
    rows = conn.execute(
        """SELECT gbr_prediction_minutes, baseline_prediction_minutes,
                  actual_completion_minutes
           FROM jobs
           WHERE source = 'live'
             AND status = 'completed'
             AND actual_completion_minutes IS NOT NULL"""
    ).fetchall()
    conn.close()

    if len(rows) < 3:
        return None

    import numpy as np
    gbr_preds = np.array([r["gbr_prediction_minutes"] for r in rows])
    baseline_preds = np.array([r["baseline_prediction_minutes"] for r in rows])
    actuals = np.array([r["actual_completion_minutes"] for r in rows])

    gbr_errors = actuals - gbr_preds
    baseline_errors = actuals - baseline_preds

    return {
        "n_records": len(rows),
        "gbr_mae": float(np.mean(np.abs(gbr_errors))),
        "gbr_mean_error": float(np.mean(gbr_errors)),
        "baseline_mae": float(np.mean(np.abs(baseline_errors))),
        "baseline_mean_error": float(np.mean(baseline_errors)),
        "note": "Computed from live operational records only, not synthetic training data.",
    }


def get_service_type_summary(db_path: Path = None) -> list[dict]:
    """Return per-service-type stats from live completed jobs."""
    conn = ensure_db(db_path)
    rows = conn.execute(
        """SELECT service_type,
                  COUNT(*) as count,
                  AVG(actual_completion_minutes) as avg_actual,
                  AVG(gbr_prediction_minutes) as avg_predicted
           FROM jobs
           WHERE source = 'live'
             AND status = 'completed'
             AND actual_completion_minutes IS NOT NULL
           GROUP BY service_type
           ORDER BY count DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
