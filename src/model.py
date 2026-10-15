"""
model.py

Wraps preprocessing + Gradient Boosting Regression into single sklearn
Pipelines, as originally proposed for this project.

Phase 2 changes:
  - Added build_quantile_pipelines() for data-driven prediction intervals
    using GradientBoostingRegressor with loss='quantile'.
  - The quantile approach trains separate models for the 10th and 90th
    percentiles, producing an 80% prediction interval grounded in the
    training data distribution rather than a fixed ±20 minute margin.
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


def build_quantile_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    """Build quantile GBR pipelines for the 10th and 90th percentiles.

    These produce the lower and upper bounds of an 80% prediction interval.
    The interval is data-driven: the model learns from the training data's
    conditional distribution of durations, rather than using a fixed offset.

    Returns a dict with keys 'lower' (alpha=0.10) and 'upper' (alpha=0.90).
    """
    pipelines = {}
    for label, alpha in [("lower", 0.10), ("upper", 0.90)]:
        pipelines[label] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=alpha,
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return pipelines
