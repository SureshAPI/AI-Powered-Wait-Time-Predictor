# AI/ML Methodology

## Dataset
See `data/schema.md`. **All results below are computed on the synthetic
dataset (`data/synthetic_service_data.csv`, 600 rows, seed=42) and demonstrate
the technical pipeline only. They are not evidence of real-world predictive
accuracy** — that would require real historical service records, which this
project does not yet have.

## Preprocessing
`src/preprocessing.py`: categorical features (`service_type`, `phone_brand`,
`parts_availability`) and the boolean `is_warranty_case` are one-hot encoded
(`handle_unknown="ignore"` so unseen categories at prediction time don't crash
the pipeline); numeric features (`current_workload`,
`technician_availability`) pass through unchanged.

## Baseline estimator
`src/baseline.py`: predicts the mean historical duration for the job's
`service_type`, then applies a small linear adjustment for
`current_workload` and `technician_availability` fit via least squares. This
exists specifically so the Gradient Boosting model's performance is compared
against something simple and explainable, not assumed to be good.

## Gradient Boosting Regression model
`src/model.py`: `sklearn.ensemble.GradientBoostingRegressor`
(`n_estimators=200`, `max_depth=3`, `learning_rate=0.05`), wrapped with the
preprocessor in a single `sklearn.Pipeline`. Trained via `src/train.py`
(`python -m src.train`), 80/20 train/test split, `random_state=42`.

## Evaluation methodology
Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² on the held-
out 20% split.

### Phase 1 Baseline Benchmark (Historical — 600 synthetic rows)

| Model | MAE (min) | RMSE (min) | R² |
|---|---:|---:|---:|
| Baseline (service-type mean + adjustment) | 50.74 | 63.30 | 0.043 |
| Gradient Boosting Regression | 16.54 | 24.76 | 0.854 |

### Phase 2 Evaluation (Current — 800 synthetic rows with `reported_issue` & complexity)

Evaluated via `python -m src.train` (80/20 train/test split, seed=42):

| Model | MAE (min) | RMSE (min) | R² |
|---|---:|---:|---:|
| Baseline (service-type mean + adjustment) | 56.21 | 72.31 | 0.010 |
| Gradient Boosting Regression | 23.06 | 38.00 | 0.727 |

### Phase 2 Data-Driven Prediction Intervals (Quantile Gradient Boosting)

To replace the heuristic ±20-minute interval from Phase 1, Phase 2 implements
two dedicated Quantile Gradient Boosting Regressors (`loss='quantile'`, with
$\alpha=0.10$ and $\alpha=0.90$) to construct an empirical 80% prediction interval.

**Held-out evaluation metrics (160 test rows):**
- **Interval Target Coverage:** 80.0% nominal
- **Empirical Coverage on Test Set:** **73.8%** (within practical bounds given sample size and non-linear noise)
- **Average Interval Width:** **61.6 minutes**
- **Median Interval Width:** **29.0 minutes**

The median interval width of 29.0 minutes reflects tight, confident intervals for
routine repairs with parts in stock, while the wider mean (61.6 minutes) accurately
captures high-uncertainty scenarios (such as water damage diagnostic or parts on
order).

## What real-world validation would require
Real historical service records (service type, phone details, workload,
technician availability, parts status, actual duration) from the shop, ideally
covering enough volume and variety to retrain and re-evaluate the model
honestly. Until then, this pipeline is a **technical demonstration**, not a
validated predictor.

