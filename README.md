# AI-Powered Waiting Time Prediction for Mobile Repair and Warranty Services

**Project Better Tomorrow — Continuation Track (Pathway A)**
Continuation of *C29 – Design Thinking for AI-Powered Problem Solving*.

> Status: 🟡 Prototype in progress — see [Current Status](#current-status).

## Problem

Customers at mobile/electronics repair and warranty service shops do not know how long
their repair or service will take. Wait time varies with repair type, phone brand/model,
reported issue, current workload, technician availability, warranty procedure, and parts
availability. This uncertainty interferes with customers' work, college, travel, and other
schedules.

This project originated from a real field visit conducted at a mobile/electronics repair
shop in Gandhipuram, Coimbatore, during the previous Design Thinking phase (Empathise →
Define). This continuation begins at **Ideate → Prototype → Validate → Deliver** — the
Empathise and Define stages are not being repeated.

## Proposed Solution

An AI-assisted system that takes job details (service type, phone brand/model, reported
issue, current workload, technician availability) and produces an **estimated service/
waiting-time range** — communicated explicitly as an estimate, never as a guarantee.

## Objectives

- Give customers a realistic waiting-time estimate before they decide to wait.
- Give employees a simple tool to enter job details and share an estimate.
- Demonstrate a technically honest AI/ML pipeline (baseline estimator + Gradient Boosting
  Regression) without overstating predictive power on limited data.
- Produce evidence (prototype + real user testing with ≥3 testers) suitable for academic
  validation.

## Features

_Updated as implementation progresses — see [Current Status](#current-status)._

## Architecture

See [`docs/architecture.md`](docs/architecture.md).

## AI / ML Approach

See [`docs/methodology.md`](docs/methodology.md) for dataset, features, preprocessing,
model choice, and evaluation methodology. **All training data used before real service
records are collected is synthetic and is explicitly labelled as such** — it is used to
demonstrate the technical pipeline, not to claim real-world predictive accuracy.

## Prototype

```bash
pip install -r requirements.txt
python src/generate_synthetic_data.py   # regenerate the synthetic dataset
python -m src.train                     # (optional) train baseline + GBR standalone, print metrics
python -m pytest tests/ -v              # run the automated tests

python app/app.py                       # run the prototype web app
# then open http://127.0.0.1:5000/
```

The app trains the model itself at startup (fast — 600 synthetic rows), so no
separate model file needs to be committed. Enter job details on the intake
form to see the estimated service-time range.

## Testing

Methodology: see [`docs/testing.md`](docs/testing.md) (added once the prototype is
testable). No user-testing results are reported until real testing has been performed.

## Responsible AI

See [`docs/responsible-ai.md`](docs/responsible-ai.md).

## Current Status

**Completed**
- Repository structure initialized.
- Problem context and continuation-track justification documented.
- Synthetic dataset schema and generator (`src/generate_synthetic_data.py`,
  `data/schema.md`) — 600 rows, clearly labelled synthetic.
- Baseline estimator (`src/baseline.py`) and Gradient Boosting Regression
  model (`src/model.py`), trained via `src/train.py`, with automated tests
  (`tests/test_pipeline.py`, 5/5 passing). Results in `docs/methodology.md`,
  labelled synthetic-data-only.
- Flask prototype app (`app/app.py`): customer intake form → prediction
  results page with an estimated range and an explicit "this is an estimate,
  not a guarantee" disclaimer. Verified working end-to-end (GET `/` and POST
  `/predict` both return 200 and render correctly).
- Employee dashboard (`app/templates/dashboard.html`) and feedback-loop job
  log (`app/job_store.py`): every prediction is logged, and an employee can
  record the real completion time for a job. Uncertainty/explanation text is
  shown to both the customer and the dashboard. Full predict → dashboard →
  record-actual flow tested end-to-end (`tests/test_pipeline.py`, 6/6
  passing).

**Planned / Future Work**
- Internal QA pass (edge cases, terminology, messaging review).
- Employee dashboard.
- User testing (3+ testers) and iteration.
- Final Prototype & Validation Report.

## Limitations

To be documented honestly as the prototype develops (data availability, model reliability
given limited/synthetic data, scope boundaries).

## Future Improvements

To be added following real user testing.
