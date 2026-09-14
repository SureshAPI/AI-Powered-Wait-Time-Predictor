# AI/ML Methodology

## Status
Planned. Will be filled in with actual dataset, feature, and evaluation
details as the pipeline is implemented (see `README.md` Current Status).

## Principles this project follows

- **No fabricated metrics.** Accuracy, MAE, RMSE, R², or any similar metric
  will only be reported if actually calculated from the stated dataset.
- **Synthetic vs. real data is always distinguished.** Any table, chart, or
  claim built on synthetic data is labelled `Synthetic / Demonstration Data`.
  Real-world validation results (from actual user testing) are labelled
  separately and only added once real results exist.
- **Technical demonstration ≠ real-world validation.** The ML pipeline can
  demonstrate that data flows correctly from intake → preprocessing → model →
  prediction without that being evidence of real-world predictive accuracy.

## Planned content (to be completed during implementation)

- Dataset description (synthetic generation logic, feature list, target).
- Preprocessing steps.
- Baseline estimator description (e.g. simple rule/average-based estimate).
- Gradient Boosting Regression model configuration.
- Evaluation methodology and (once computed) actual metrics on the synthetic
  dataset, explicitly labelled as such.
- Discussion of what would be needed for real-world predictive validity
  (i.e., real historical service records).
