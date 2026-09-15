"""
app.py

Prototype web application for Project Better Tomorrow: a customer/employee
intake form that produces an estimated service/waiting time.

The model is trained once at startup on the SYNTHETIC dataset (fast — ~600
rows). This is a prototype demonstrating the pipeline end-to-end; it is not
connected to any real shop data. See docs/methodology.md and
data/README.md.

Run with:
    python app/app.py
Then open http://127.0.0.1:5000/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from flask import Flask, abort, redirect, render_template, request, url_for

from src.baseline import BaselineEstimator
from src.model import build_gbr_pipeline
from src.preprocessing import get_features_and_target, load_dataset

from app.job_store import add_job, get_job, load_jobs, record_actual_completion

app = Flask(__name__)
app.jinja_env.filters["format_minutes"] = lambda m: format_minutes(float(m))

# --- Train once at startup on the synthetic dataset ---
_df = load_dataset()
_X, _y = get_features_and_target(_df)

_gbr = build_gbr_pipeline()
_gbr.fit(_X, _y)

_baseline = BaselineEstimator().fit(_X, _y)

# Rough uncertainty margin for the displayed range. This is a simple
# constant derived from the model's held-out RMSE reported in
# docs/methodology.md — a future iteration could replace this with proper
# quantile regression for a statistically grounded interval.
UNCERTAINTY_MARGIN_MINUTES = 20

# QA finding: the raw GBR prediction can occasionally go to zero or negative
# for very fast/low-workload jobs, since nothing in training constrains it to
# be positive. A real service can't take negative time, so clamp to a small
# practical floor (matches the floor already used in the baseline estimator).
MIN_PREDICTION_MINUTES = 5.0

SERVICE_TYPES = sorted(_df["service_type"].unique())
PHONE_BRANDS = sorted(_df["phone_brand"].unique())
PARTS_OPTIONS = sorted(_df["parts_availability"].unique())


def format_minutes(minutes: float) -> str:
    """QA finding: raw minute counts (e.g. '229 minutes') are hard to read
    for longer jobs. Format as hours + minutes once over an hour."""
    minutes = round(minutes)
    if minutes < 60:
        return f"{minutes} min"
    hours, rem = divmod(minutes, 60)
    return f"{hours} hr {rem} min" if rem else f"{hours} hr"


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        service_types=SERVICE_TYPES,
        phone_brands=PHONE_BRANDS,
        parts_options=PARTS_OPTIONS,
    )


def status_note_for(inputs: dict) -> str:
    """Short, honest context line for the result board — based on the
    actual inputs, not invented."""
    if inputs["parts_availability"] == "Order Required":
        return "Parts need to be ordered for this job — this can extend the wait."
    if inputs["technician_availability"] == 0:
        return "No technician is currently free — the wait may run longer than usual."
    return "Estimate based on current shop workload and technician availability."


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    inputs = {
        "service_type": form["service_type"],
        "phone_brand": form["phone_brand"],
        "is_warranty_case": form.get("is_warranty_case") == "on",
        "parts_availability": form["parts_availability"],
        "current_workload": int(form["current_workload"]),
        "technician_availability": int(form["technician_availability"]),
    }
    row = pd.DataFrame([inputs])

    gbr_pred = float(_gbr.predict(row)[0])
    gbr_pred = max(gbr_pred, MIN_PREDICTION_MINUTES)  # QA fix: no negative/near-zero estimates
    baseline_pred = float(_baseline.predict(row)[0])

    job_id = add_job(inputs, gbr_pred, baseline_pred)

    # Robustness fix: redirect to a real GET URL for the result (Post/Redirect/
    # Get) instead of rendering directly on the POST response, so refreshing
    # or bookmarking the result page works instead of showing "Method Not
    # Allowed".
    return redirect(url_for("result", job_id=job_id))


@app.route("/result/<job_id>", methods=["GET"])
def result(job_id):
    job = get_job(job_id)
    if job is None:
        abort(404)

    inputs = {
        "service_type": job["service_type"],
        "phone_brand": job["phone_brand"],
        "is_warranty_case": job["is_warranty_case"] == "True",
        "parts_availability": job["parts_availability"],
        "current_workload": int(job["current_workload"]),
        "technician_availability": int(job["technician_availability"]),
    }
    gbr_pred = float(job["gbr_prediction_minutes"])

    low = max(gbr_pred - UNCERTAINTY_MARGIN_MINUTES, MIN_PREDICTION_MINUTES)
    high = gbr_pred + UNCERTAINTY_MARGIN_MINUTES

    return render_template(
        "result.html",
        job_id=job_id,
        estimate=format_minutes(gbr_pred),
        low=format_minutes(low),
        high=format_minutes(high),
        inputs=inputs,
        status_note=status_note_for(inputs),
    )


@app.route("/dashboard", methods=["GET"])
def dashboard():
    jobs = load_jobs()
    open_count = sum(1 for j in jobs if j["status"] == "open")
    completed_count = sum(1 for j in jobs if j["status"] == "completed")
    return render_template(
        "dashboard.html",
        jobs=jobs,
        open_count=open_count,
        completed_count=completed_count,
    )


@app.route("/dashboard/complete/<job_id>", methods=["POST"])
def complete_job(job_id):
    actual_minutes = float(request.form["actual_completion_minutes"])
    record_actual_completion(job_id, actual_minutes)
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=True)
