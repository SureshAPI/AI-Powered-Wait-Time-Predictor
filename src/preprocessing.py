"""
preprocessing.py

Loads the (synthetic) service dataset and builds the preprocessing pipeline
shared by the baseline estimator and the Gradient Boosting model.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_service_data.csv"

CATEGORICAL_FEATURES = ["service_type", "phone_brand", "parts_availability"]
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
