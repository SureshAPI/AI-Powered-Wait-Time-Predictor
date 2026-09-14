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
from flask import Flask, render_template, request

from src.baseline import BaselineEstimator
from src.model import build_gbr_pipeline
from src.preprocessing import get_features_and_target, load_dataset

app = Flask(__name__)

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

SERVICE_TYPES = sorted(_df["service_type"].unique())
PHONE_BRANDS = sorted(_df["phone_brand"].unique())
PARTS_OPTIONS = sorted(_df["parts_availability"].unique())


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        service_types=SERVICE_TYPES,
        phone_brands=PHONE_BRANDS,
        parts_options=PARTS_OPTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    row = pd.DataFrame(
        [
            {
                "service_type": form["service_type"],
                "phone_brand": form["phone_brand"],
                "is_warranty_case": form.get("is_warranty_case") == "on",
                "parts_availability": form["parts_availability"],
                "current_workload": int(form["current_workload"]),
                "technician_availability": int(form["technician_availability"]),
            }
        ]
    )

    gbr_pred = float(_gbr.predict(row)[0])
    baseline_pred = float(_baseline.predict(row)[0])  # kept for the future employee dashboard

    low = max(gbr_pred - UNCERTAINTY_MARGIN_MINUTES, 5)
    high = gbr_pred + UNCERTAINTY_MARGIN_MINUTES

    return render_template(
        "result.html",
        estimate=round(gbr_pred),
        low=round(low),
        high=round(high),
        inputs=row.iloc[0].to_dict(),
    )


if __name__ == "__main__":
    app.run(debug=True)
