"""
preprocessing.py

Loads the (synthetic) service dataset and builds the preprocessing pipeline
shared by the baseline estimator and the Gradient Boosting model.

Phase 2 changes:
  - Added reported_issue to CATEGORICAL_FEATURES.
  - Added input validation helpers for the web app.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_service_data.csv"

CATEGORICAL_FEATURES = ["service_type", "reported_issue", "phone_brand", "parts_availability"]
BOOLEAN_FEATURES = ["is_warranty_case"]
NUMERIC_FEATURES = ["current_workload", "technician_availability"]
TARGET = "actual_duration_minutes"

FEATURE_COLUMNS = CATEGORICAL_FEATURES + BOOLEAN_FEATURES + NUMERIC_FEATURES


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset from CSV. Raises a clear error if it hasn't been
    generated yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run "
            "`python src/generate_synthetic_data.py` first."
        )
    df = pd.read_csv(path)
    df["is_warranty_case"] = df["is_warranty_case"].astype(bool)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Column transformer: one-hot encode categoricals + booleans,
    pass numeric features through unchanged."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES + BOOLEAN_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def get_features_and_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    return X, y


# ---------------------------------------------------------------------------
# Phase 2: Input validation helpers
# ---------------------------------------------------------------------------

def validate_prediction_inputs(form_data: dict) -> tuple[dict | None, list[str]]:
    """Validate and parse intake form data.

    Returns (parsed_inputs, errors). If errors is non-empty, parsed_inputs
    is None and errors contains user-facing messages.
    """
    errors = []

    # Import here to avoid circular imports at module level
    from src.generate_synthetic_data import (
        PARTS_AVAILABILITY,
        PHONE_BRANDS,
        REPORTED_ISSUES,
        SERVICE_TYPES,
    )

    service_type = form_data.get("service_type", "").strip()
    if not service_type:
        errors.append("Service type is required.")
    elif service_type not in SERVICE_TYPES:
        errors.append(f"Unknown service type: {service_type}")

    reported_issue = form_data.get("reported_issue", "").strip()
    if not reported_issue:
        errors.append("Reported issue is required.")
    elif service_type in REPORTED_ISSUES and reported_issue not in REPORTED_ISSUES[service_type]:
        errors.append(f"Invalid reported issue for {service_type}: {reported_issue}")

    phone_brand = form_data.get("phone_brand", "").strip()
    if not phone_brand:
        errors.append("Phone brand is required.")
    elif phone_brand not in PHONE_BRANDS:
        errors.append(f"Unknown phone brand: {phone_brand}")

    parts_availability = form_data.get("parts_availability", "").strip()
    if not parts_availability:
        errors.append("Parts availability is required.")
    elif parts_availability not in PARTS_AVAILABILITY:
        errors.append(f"Unknown parts availability: {parts_availability}")

    # Numeric fields
    try:
        current_workload = int(form_data.get("current_workload", ""))
        if current_workload < 0 or current_workload > 30:
            errors.append("Current workload must be between 0 and 30.")
    except (ValueError, TypeError):
        errors.append("Current workload must be a valid integer.")
        current_workload = None

    try:
        technician_availability = int(form_data.get("technician_availability", ""))
        if technician_availability < 0 or technician_availability > 10:
            errors.append("Technician availability must be between 0 and 10.")
    except (ValueError, TypeError):
        errors.append("Technician availability must be a valid integer.")
        technician_availability = None

    is_warranty_case = form_data.get("is_warranty_case") == "on"

    if errors:
        return None, errors

    return {
        "service_type": service_type,
        "reported_issue": reported_issue,
        "phone_brand": phone_brand,
        "is_warranty_case": is_warranty_case,
        "parts_availability": parts_availability,
        "current_workload": current_workload,
        "technician_availability": technician_availability,
    }, []
