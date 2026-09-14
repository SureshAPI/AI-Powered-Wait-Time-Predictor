"""
train.py

Trains the baseline estimator and the Gradient Boosting Regression model on
the SYNTHETIC dataset, evaluates both on a held-out split, and saves the GBR
model artifact. Prints metrics clearly labelled as computed on synthetic
data — these are NOT real-world validation results.

Usage:
    python -m src.train
"""

from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.baseline import BaselineEstimator
from src.model import build_gbr_pipeline
from src.preprocessing import get_features_and_target, load_dataset

MODEL_OUT_PATH = Path(__file__).resolve().parent.parent / "models" / "gbr_model.joblib"


def evaluate(name: str, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"[SYNTHETIC DATA ONLY] {name}: MAE={mae:.2f} min, RMSE={rmse:.2f} min, R2={r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    df = load_dataset()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=" * 70)
    print("Training on SYNTHETIC / DEMONSTRATION data only.")
    print("These metrics demonstrate the pipeline, not real-world accuracy.")
    print("=" * 70)

    baseline = BaselineEstimator().fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    evaluate("Baseline (service-type mean + workload/technician adjustment)", y_test, baseline_preds)

    gbr = build_gbr_pipeline()
    gbr.fit(X_train, y_train)
    gbr_preds = gbr.predict(X_test)
    evaluate("Gradient Boosting Regression", y_test, gbr_preds)

    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gbr, MODEL_OUT_PATH)
    print(f"Saved trained GBR pipeline to {MODEL_OUT_PATH} (not committed to git; regenerate via this script).")


if __name__ == "__main__":
    main()
