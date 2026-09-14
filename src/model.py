"""
model.py

Wraps preprocessing + Gradient Boosting Regression into a single sklearn
Pipeline, as originally proposed for this project.
"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def build_gbr_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=random_state,
                ),
            ),
        ]
    )
