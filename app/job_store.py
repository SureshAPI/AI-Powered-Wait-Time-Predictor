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

Phase 3 additions:
  - job_events table for service timeline tracking.
  - Event logging in add_job(), update_job_status().
  - Customer ticket tracking (get_job_by_ticket_code).
  - CSV export (export_jobs_csv).

The database file lives at data/jobs.db (git-ignored). Legacy CSV records
in data/job_log.csv are migrated once on first access and then the CSV
is left in place (not deleted) for safety.
"""

import csv
import io
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

# Phase 3: Event timeline table
_CREATE_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    details TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);
"""

_CREATE_EVENTS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_events_job_id ON job_events(job_id);"
)


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
    conn.execute(_CREATE_EVENTS_TABLE_SQL)
    conn.execute(_CREATE_EVENTS_INDEX_SQL)
    conn.commit()


def add_event(job_id: str, event_type: str, details: str = None,
              timestamp: str = None, conn: sqlite3.Connection = None,
              db_path: Path = None) -> int:
    """Record a timeline event for a job (Phase 3)."""
    close_after = False
    if conn is None:
        conn = ensure_db(db_path)
        close_after = True
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO job_events (job_id, event_type, timestamp, details) VALUES (?, ?, ?, ?)",
        (job_id, event_type, timestamp, details),
    )
    conn.commit()
    event_id = cur.lastrowid
    if close_after:
        conn.close()
    return event_id


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
    # Phase 3: Log creation event in timeline
    service_desc = inputs["service_type"]
    if inputs.get("reported_issue"):
        service_desc += f" ({inputs['reported_issue']})"
    add_event(
        job_id=job_id,
        event_type="created",
        details=f"Intake registered: {service_desc}. Estimated wait: {round(gbr_prediction, 1)} min.",
        timestamp=now,
        conn=conn,
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
        add_event(
            job_id=job_id,
            event_type="completed",
            details=f"Service completed. Actual duration recorded: {round(actual_minutes, 1)} min.",
            timestamp=now,
            conn=conn,
        )
    else:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
            (new_status, now, job_id),
        )
        event_descriptions = {
            "in_progress": "Technician started diagnostic and repair work.",
            "cancelled": "Service request cancelled.",
            "open": "Service request reopened.",
        }
        add_event(
            job_id=job_id,
            event_type=new_status,
            details=event_descriptions.get(new_status, f"Status updated to {new_status}."),
            timestamp=now,
            conn=conn,
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


# ---------------------------------------------------------------------------
# Phase 3: Timeline, Customer Tracking, CSV Export & Workload Intelligence
# ---------------------------------------------------------------------------

def get_job_events(job_id: str, db_path: Path = None) -> list[dict]:
    """Retrieve all chronological events for a job.
    
    If no events are logged yet (e.g. for legacy or demo jobs), constructs
    sensible fallback events from the job row so the timeline always renders.
    """
    conn = ensure_db(db_path)
    rows = conn.execute(
        "SELECT * FROM job_events WHERE job_id = ? ORDER BY timestamp ASC, id ASC",
        (job_id,),
    ).fetchall()
    events = [dict(r) for r in rows]
    
    if not events:
        job = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if job:
            events.append({
                "id": 0,
                "job_id": job_id,
                "event_type": "created",
                "timestamp": job["created_at"],
                "details": f"Job registered in system ({job['service_type']}). Initial AI estimate: {job['gbr_prediction_minutes']} min.",
            })
            if job["status"] in ("in_progress", "completed"):
                events.append({
                    "id": 1,
                    "job_id": job_id,
                    "event_type": "in_progress",
                    "timestamp": job["updated_at"] or job["created_at"],
                    "details": "Technician started service.",
                })
            if job["status"] == "completed":
                actual_min = job["actual_completion_minutes"]
                details = f"Service completed. Actual duration: {actual_min} min." if actual_min else "Service completed."
                events.append({
                    "id": 2,
                    "job_id": job_id,
                    "event_type": "completed",
                    "timestamp": job["updated_at"] or job["created_at"],
                    "details": details,
                })
            elif job["status"] == "cancelled":
                events.append({
                    "id": 3,
                    "job_id": job_id,
                    "event_type": "cancelled",
                    "timestamp": job["updated_at"] or job["created_at"],
                    "details": "Service request cancelled.",
                })

    conn.close()
    return events


def get_job_by_ticket_code(ticket_code: str, db_path: Path = None) -> dict | None:
    """Case-insensitive lookup of a job by ticket reference code."""
    if not ticket_code:
        return None
    code = str(ticket_code).strip()
    conn = ensure_db(db_path)
    row = conn.execute(
        "SELECT * FROM jobs WHERE LOWER(job_id) = LOWER(?)",
        (code,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def export_jobs_csv(status_filter: str = None, search_query: str = None,
                    service_type_filter: str = None, source_filter: str = None,
                    db_path: Path = None) -> str:
    """Generate a RFC-4180 compliant CSV string of jobs matching filters."""
    jobs = load_jobs(
        status_filter=status_filter,
        search_query=search_query,
        service_type_filter=service_type_filter,
        source_filter=source_filter,
        db_path=db_path,
    )
    output = io.StringIO()
    fieldnames = [
        "job_id",
        "created_at",
        "status",
        "service_type",
        "reported_issue",
        "phone_brand",
        "is_warranty_case",
        "parts_availability",
        "current_workload",
        "technician_availability",
        "gbr_prediction_minutes",
        "interval_lower_minutes",
        "interval_upper_minutes",
        "baseline_prediction_minutes",
        "actual_completion_minutes",
        "source",
        "updated_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for j in jobs:
        writer.writerow(j)
    return output.getvalue()


def check_job_delay(job: dict) -> dict:
    """Evaluate whether an active job has exceeded its predicted waiting time.
    
    Returns a dict with 'is_delayed', 'elapsed_minutes', and 'delay_minutes'.
    """
    created_at_str = job.get("created_at")
    if not created_at_str or job.get("status") in ("completed", "cancelled"):
        return {"is_delayed": False, "elapsed_minutes": 0.0, "delay_minutes": 0.0}

    try:
        created_dt = datetime.fromisoformat(created_at_str)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        elapsed_minutes = max(0.0, (now - created_dt).total_seconds() / 60.0)
    except Exception:
        return {"is_delayed": False, "elapsed_minutes": 0.0, "delay_minutes": 0.0}

    pred_minutes = float(job.get("gbr_prediction_minutes") or 0.0)
    is_delayed = elapsed_minutes > pred_minutes and pred_minutes > 0
    delay_minutes = max(0.0, elapsed_minutes - pred_minutes)
    return {
        "is_delayed": is_delayed,
        "elapsed_minutes": round(elapsed_minutes, 1),
        "delay_minutes": round(delay_minutes, 1),
    }


def get_workload_intelligence(db_path: Path = None) -> dict:
    """Compute queue management insights and workload metrics for employees."""
    conn = ensure_db(db_path)
    rows = conn.execute(
        "SELECT * FROM jobs WHERE status IN ('open', 'in_progress') ORDER BY created_at ASC"
    ).fetchall()
    conn.close()

    active_jobs = [dict(r) for r in rows]
    open_jobs = [j for j in active_jobs if j["status"] == "open"]
    in_progress_jobs = [j for j in active_jobs if j["status"] == "in_progress"]

    total_backlog_minutes = sum(float(j.get("gbr_prediction_minutes") or 0.0) for j in active_jobs)

    overdue_count = 0
    high_uncertainty_count = 0
    for j in active_jobs:
        delay_info = check_job_delay(j)
        if delay_info["is_delayed"]:
            overdue_count += 1
        low = j.get("interval_lower_minutes")
        high = j.get("interval_upper_minutes")
        if (low is not None and high is not None and (float(high) - float(low)) >= 45.0) or j.get("parts_availability") == "Order Required":
            high_uncertainty_count += 1

    return {
        "active_count": len(active_jobs),
        "open_count": len(open_jobs),
        "in_progress_count": len(in_progress_jobs),
        "total_backlog_minutes": round(total_backlog_minutes, 1),
        "overdue_count": overdue_count,
        "high_uncertainty_count": high_uncertainty_count,
    }

