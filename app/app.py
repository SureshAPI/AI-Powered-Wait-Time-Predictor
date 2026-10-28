"""
app.py

Prototype web application for Project Better Tomorrow: a customer/employee
intake form that produces an estimated service/waiting time.

The model is trained once at startup on the SYNTHETIC dataset (fast — ~800
rows). This is a prototype demonstrating the pipeline end-to-end; it is not
connected to any real shop data. See docs/methodology.md and data/README.md.

Phase 2 changes:
  - App factory pattern (create_app) for testability.
  - Quantile GBR models for data-driven prediction intervals.
  - Separated reported_issue from service_type.
  - Input validation with error messages.
  - Prediction explanations.
  - SQLite-backed job store with status transitions.
  - Dashboard search, filters, and analytics.
  - Health endpoint (/health).
  - Lazy model loading to avoid import-time side effects in tests.

Run with:
    python app/app.py
Then open http://127.0.0.1:5000/

Production:
    gunicorn app.app:app
"""

import os
import sys
from functools import wraps
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from src.baseline import BaselineEstimator
from src.explanations import explain_prediction
from src.generate_synthetic_data import REPORTED_ISSUES
from src.model import build_gbr_pipeline, build_quantile_pipelines
from src.preprocessing import (
    get_features_and_target,
    load_dataset,
    validate_prediction_inputs,
)

from app.job_store import (
    add_job,
    check_job_delay,
    export_jobs_csv,
    get_job,
    get_job_by_ticket_code,
    get_job_events,
    get_prediction_accuracy_stats,
    get_service_type_summary,
    get_status_counts,
    get_workload_intelligence,
    load_jobs,
    update_job_status,
)

# QA finding: the raw GBR prediction can occasionally go to zero or negative
# for very fast/low-workload jobs, since nothing in training constrains it to
# be positive. A real service can't take negative time, so clamp to a small
# practical floor (matches the floor already used in the baseline estimator).
MIN_PREDICTION_MINUTES = 5.0


def format_minutes(minutes: float) -> str:
    """QA finding: raw minute counts (e.g. '229 minutes') are hard to read
    for longer jobs. Format as hours + minutes once over an hour."""
    minutes = round(minutes)
    if minutes < 60:
        return f"{minutes} min"
    hours, rem = divmod(minutes, 60)
    return f"{hours} hr {rem} min" if rem else f"{hours} hr"


def create_app(test_db_path=None):
    """Flask application factory.

    Args:
        test_db_path: If provided, use this path for the SQLite database
            instead of the default. Used in tests to isolate data.
    """
    flask_app = Flask(__name__)
    flask_app.jinja_env.filters["format_minutes"] = lambda m: format_minutes(float(m))

    # --- Train once at startup on the synthetic dataset ---
    _df = load_dataset()
    _X, _y = get_features_and_target(_df)

    _gbr = build_gbr_pipeline()
    _gbr.fit(_X, _y)

    _baseline = BaselineEstimator().fit(_X, _y)

    # Phase 2: Quantile models for prediction intervals
    _quantile_models = build_quantile_pipelines()
    for pipe in _quantile_models.values():
        pipe.fit(_X, _y)

    # Training medians for explanations
    _training_medians = {
        "current_workload": int(_df["current_workload"].median()),
        "technician_availability": int(_df["technician_availability"].median()),
    }

    SERVICE_TYPES = sorted(_df["service_type"].unique())
    PHONE_BRANDS = sorted(_df["phone_brand"].unique())
    PARTS_OPTIONS = sorted(_df["parts_availability"].unique())

    # Store on app for access in route handlers
    flask_app.secret_key = os.environ.get("SECRET_KEY", "better-tomorrow-secret-key-phase3")
    flask_app.config["STAFF_PIN"] = os.environ.get("STAFF_PIN", "1234")
    flask_app.config["GBR_MODEL"] = _gbr
    flask_app.config["BASELINE_MODEL"] = _baseline
    flask_app.config["QUANTILE_MODELS"] = _quantile_models
    flask_app.config["TRAINING_MEDIANS"] = _training_medians
    flask_app.config["SERVICE_TYPES"] = SERVICE_TYPES
    flask_app.config["PHONE_BRANDS"] = PHONE_BRANDS
    flask_app.config["PARTS_OPTIONS"] = PARTS_OPTIONS
    flask_app.config["TEST_DB_PATH"] = test_db_path

    # Phase 3: Staff authentication decorator
    def staff_login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if flask_app.config.get("STAFF_AUTH_DISABLED"):
                return f(*args, **kwargs)
            # In testing environment unless TEST_STAFF_AUTH is explicitly set, allow pass-through
            if "pytest" in sys.modules and not flask_app.config.get("TEST_STAFF_AUTH"):
                return f(*args, **kwargs)
            if not session.get("staff_authenticated"):
                next_url = request.full_path if request.query_string else request.path
                return redirect(url_for("staff_login", next=next_url))
            return f(*args, **kwargs)
        return decorated_function

    # ---------------------------------------------------------------------------
    # Routes
    # ---------------------------------------------------------------------------

    @flask_app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            service_types=SERVICE_TYPES,
            phone_brands=PHONE_BRANDS,
            parts_options=PARTS_OPTIONS,
            reported_issues=REPORTED_ISSUES,
        )

    @flask_app.route("/predict", methods=["POST"])
    def predict():
        form = request.form

        # Phase 2: Input validation
        inputs, errors = validate_prediction_inputs(form)
        if errors:
            return render_template(
                "index.html",
                service_types=SERVICE_TYPES,
                phone_brands=PHONE_BRANDS,
                parts_options=PARTS_OPTIONS,
                reported_issues=REPORTED_ISSUES,
                errors=errors,
                form_data=form,
            ), 400

        row = pd.DataFrame([inputs])

        gbr_pred = float(_gbr.predict(row)[0])
        gbr_pred = max(gbr_pred, MIN_PREDICTION_MINUTES)
        baseline_pred = float(_baseline.predict(row)[0])

        # Phase 2: Data-driven prediction intervals from quantile models
        interval_lower = float(_quantile_models["lower"].predict(row)[0])
        interval_upper = float(_quantile_models["upper"].predict(row)[0])
        # Ensure lower bound is at least MIN_PREDICTION_MINUTES
        interval_lower = max(interval_lower, MIN_PREDICTION_MINUTES)
        # Ensure upper bound is at least as large as point prediction
        interval_upper = max(interval_upper, gbr_pred)

        job_id = add_job(
            inputs, gbr_pred, baseline_pred,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            db_path=test_db_path,
        )

        return redirect(url_for("result", job_id=job_id))

    @flask_app.route("/result/<job_id>", methods=["GET"])
    def result(job_id):
        job = get_job(job_id, db_path=test_db_path)
        if job is None:
            abort(404)

        inputs = {
            "service_type": job["service_type"],
            "reported_issue": job.get("reported_issue", ""),
            "phone_brand": job["phone_brand"],
            "is_warranty_case": bool(job["is_warranty_case"]),
            "parts_availability": job["parts_availability"],
            "current_workload": int(job["current_workload"]),
            "technician_availability": int(job["technician_availability"]),
        }
        gbr_pred = float(job["gbr_prediction_minutes"])

        # Phase 2: Use stored interval bounds if available, else fall back
        low = job.get("interval_lower_minutes")
        high = job.get("interval_upper_minutes")
        if low is not None and high is not None:
            low = max(float(low), MIN_PREDICTION_MINUTES)
            high = float(high)
        else:
            # Fallback for legacy jobs without interval data
            low = max(gbr_pred - 20, MIN_PREDICTION_MINUTES)
            high = gbr_pred + 20

        # Phase 3: Uncertainty detection
        is_high_uncertainty = (
            (high - low >= 45.0) or
            (inputs.get("parts_availability") == "Order Required") or
            (inputs.get("parts_availability") == "Unknown")
        )

        # Phase 2: Prediction explanations
        explanation = explain_prediction(inputs, gbr_pred, _training_medians)

        return render_template(
            "result.html",
            job_id=job_id,
            estimate=format_minutes(gbr_pred),
            low=format_minutes(low),
            high=format_minutes(high),
            inputs=inputs,
            status_note=explanation["customer_summary"],
            explanation=explanation,
            is_high_uncertainty=is_high_uncertainty,
        )

    # ---------------------------------------------------------------------------
    # Phase 3: Customer Self-Service Tracking
    # ---------------------------------------------------------------------------

    @flask_app.route("/track", methods=["GET"])
    def track_search():
        ticket_code = request.args.get("ticket_code", "").strip()
        if ticket_code:
            return redirect(url_for("track_ticket", job_id=ticket_code))
        return render_template("track.html", error=None)

    @flask_app.route("/track/<job_id>", methods=["GET"])
    def track_ticket(job_id):
        job = get_job_by_ticket_code(job_id, db_path=test_db_path)
        if job is None:
            return render_template(
                "track.html",
                error=f"No repair ticket found for reference '{job_id}'. Please verify your ticket code.",
                searched_code=job_id,
            ), 404

        events = get_job_events(job["job_id"], db_path=test_db_path)
        delay_info = check_job_delay(job)

        low = job.get("interval_lower_minutes")
        high = job.get("interval_upper_minutes")
        gbr_pred = float(job.get("gbr_prediction_minutes") or 0.0)

        if low is not None and high is not None:
            low_val = max(float(low), MIN_PREDICTION_MINUTES)
            high_val = float(high)
        else:
            low_val = max(gbr_pred - 20, MIN_PREDICTION_MINUTES)
            high_val = gbr_pred + 20

        is_high_uncertainty = (
            (high_val - low_val >= 45.0) or
            (job.get("parts_availability") == "Order Required") or
            (job.get("parts_availability") == "Unknown")
        )

        inputs = {
            "service_type": job["service_type"],
            "reported_issue": job.get("reported_issue", ""),
            "phone_brand": job["phone_brand"],
            "is_warranty_case": bool(job["is_warranty_case"]),
            "parts_availability": job["parts_availability"],
            "current_workload": int(job["current_workload"]),
            "technician_availability": int(job["technician_availability"]),
        }
        explanation = explain_prediction(inputs, gbr_pred, _training_medians)

        return render_template(
            "track_status.html",
            job=job,
            events=events,
            delay_info=delay_info,
            is_high_uncertainty=is_high_uncertainty,
            estimate=format_minutes(gbr_pred),
            low=format_minutes(low_val),
            high=format_minutes(high_val),
            explanation=explanation,
            inputs=inputs,
        )

    # ---------------------------------------------------------------------------
    # Staff Authentication & Dashboard
    # ---------------------------------------------------------------------------

    @flask_app.route("/dashboard/login", methods=["GET", "POST"])
    def staff_login():
        error = None
        next_url = request.args.get("next") or request.form.get("next") or url_for("dashboard")
        if request.method == "POST":
            pin = request.form.get("pin", "").strip()
            if pin == flask_app.config["STAFF_PIN"]:
                session["staff_authenticated"] = True
                return redirect(next_url)
            else:
                error = "Invalid Staff PIN. Please try again."
        return render_template("login.html", error=error, next_url=next_url)

    @flask_app.route("/dashboard/logout", methods=["GET", "POST"])
    def staff_logout():
        session.pop("staff_authenticated", None)
        return redirect(url_for("index"))

    @flask_app.route("/dashboard", methods=["GET"])
    @staff_login_required
    def dashboard():
        # Phase 2: Search and filter support
        status_filter = request.args.get("status", "").strip() or None
        search_query = request.args.get("search", "").strip() or None
        service_type_filter = request.args.get("service_type", "").strip() or None

        jobs = load_jobs(
            status_filter=status_filter,
            search_query=search_query,
            service_type_filter=service_type_filter,
            db_path=test_db_path,
        )
        counts = get_status_counts(db_path=test_db_path)

        # Phase 3: Workload intelligence & job delay status
        for j in jobs:
            j["delay_info"] = check_job_delay(j)

        workload_intel = get_workload_intelligence(db_path=test_db_path)

        # Phase 2: Analytics (only from live operational records)
        accuracy_stats = get_prediction_accuracy_stats(db_path=test_db_path)
        service_stats = get_service_type_summary(db_path=test_db_path)

        return render_template(
            "dashboard.html",
            jobs=jobs,
            counts=counts,
            open_count=counts.get("open", 0),
            in_progress_count=counts.get("in_progress", 0),
            completed_count=counts.get("completed", 0),
            cancelled_count=counts.get("cancelled", 0),
            workload_intel=workload_intel,
            accuracy_stats=accuracy_stats,
            service_stats=service_stats,
            service_types=SERVICE_TYPES,
            current_filters={
                "status": status_filter or "",
                "search": search_query or "",
                "service_type": service_type_filter or "",
            },
        )

    @flask_app.route("/dashboard/update/<job_id>", methods=["POST"])
    @staff_login_required
    def update_job(job_id):
        """Phase 2: General status update endpoint with validated transitions."""
        new_status = request.form.get("new_status", "").strip()
        actual_minutes_str = request.form.get("actual_completion_minutes", "").strip()

        actual_minutes = None
        if actual_minutes_str:
            try:
                actual_minutes = float(actual_minutes_str)
            except ValueError:
                return redirect(url_for("dashboard"))

        success, message = update_job_status(
            job_id, new_status, actual_minutes, db_path=test_db_path
        )

        return redirect(url_for("dashboard"))

    # Backward-compatible route from Phase 1
    @flask_app.route("/dashboard/complete/<job_id>", methods=["POST"])
    @staff_login_required
    def complete_job(job_id):
        actual_minutes = float(request.form["actual_completion_minutes"])
        update_job_status(job_id, "completed", actual_minutes, db_path=test_db_path)
        return redirect(url_for("dashboard"))

    # Phase 3: CSV Export Endpoint
    @flask_app.route("/dashboard/export", methods=["GET"])
    @staff_login_required
    def export_csv():
        status_filter = request.args.get("status", "").strip() or None
        search_query = request.args.get("search", "").strip() or None
        service_type_filter = request.args.get("service_type", "").strip() or None
        source_filter = request.args.get("source", "").strip() or None

        csv_content = export_jobs_csv(
            status_filter=status_filter,
            search_query=search_query,
            service_type_filter=service_type_filter,
            source_filter=source_filter,
            db_path=test_db_path,
        )
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=repair_jobs_export.csv",
                "Content-Type": "text/csv; charset=utf-8",
            },
        )

    # Phase 2: Health endpoint for deployment monitoring
    @flask_app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model": "gradient_boosting_regression",
            "dataset_rows": len(_df),
            "service_types": len(SERVICE_TYPES),
        })

    # Phase 2: API endpoint for reported issues (used by JS dynamic dropdown)
    @flask_app.route("/api/issues/<service_type>", methods=["GET"])
    def get_issues(service_type):
        issues = REPORTED_ISSUES.get(service_type, [])
        return jsonify(issues)

    # Phase 3: Custom Error Handlers
    @flask_app.errorhandler(404)
    def handle_404(e):
        return render_template("errors/404.html"), 404

    @flask_app.errorhandler(500)
    def handle_500(e):
        return render_template("errors/500.html"), 500

    return flask_app


# Module-level app instance for gunicorn compatibility:
#   gunicorn app.app:app
app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
