"""
baseline.py

A simple, explainable baseline estimator: predicts the average historical
duration for a given service_type, then applies a small linear adjustment for
current workload and technician availability. This exists so the Gradient
Boosting model's performance can be honestly compared against something
simpler, rather than assumed to be good.
"""

import pandas as pd


class BaselineEstimator:
    """Rule/average-based baseline: mean duration per service_type, adjusted
    for workload and technician availability using coefficients fit by
    simple linear regression on those two numeric features' residuals."""

    def __init__(self):
        self.service_type_means_: dict[str, float] = {}
        self.workload_coef_: float = 0.0
        self.technician_coef_: float = 0.0
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineEstimator":
        df = X.copy()
        df["_y"] = y.values

        self.global_mean_ = float(df["_y"].mean())
        self.service_type_means_ = (
            df.groupby("service_type")["_y"].mean().to_dict()
        )

        # Residual after removing the per-service-type mean
        df["_service_mean"] = df["service_type"].map(self.service_type_means_)
        df["_residual"] = df["_y"] - df["_service_mean"]

        # Fit simple coefficients via least squares on two numeric features
        import numpy as np

        A = np.column_stack(
            [df["current_workload"].values, df["technician_availability"].values]
        )
        coefs, *_ = np.linalg.lstsq(A, df["_residual"].values, rcond=None)
        self.workload_coef_, self.technician_coef_ = coefs
        return self

    def predict(self, X: pd.DataFrame):
        preds = []
        for _, row in X.iterrows():
            base = self.service_type_means_.get(
                row["service_type"], self.global_mean_
            )
            adjustment = (
                self.workload_coef_ * row["current_workload"]
                + self.technician_coef_ * row["technician_availability"]
            )
            preds.append(max(base + adjustment, 5.0))
        return preds
