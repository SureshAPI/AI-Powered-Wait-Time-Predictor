# System Architecture

## Status
Complete production-ready prototype (100% cumulative project completion across Phase 1, Phase 2, and Phase 3). Evaluated and verified on automated test suite (23/23 tests passing).

## Prototype Format Decision

The prototype will be a **lightweight web application** (Flask backend serving
a simple HTML/JS intake form + results page, with a scikit-learn model behind
a prediction endpoint), rather than a large full-stack system, a static
Figma/Canva mockup, or a general repair-shop management platform.

Reasoning, evaluated against Impact + Demonstrability + Technical Credibility +
Usability + Feasibility + Academic Defensibility:

- **Impact / core value proposition**: a working intake form → prediction flow
  directly demonstrates "know your expected waiting time before you wait"
  without needing to build unrelated shop-management features.
- **Demonstrability**: a tester can actually perform the task (enter job
  details, receive an estimate) rather than looking at a static mockup —
  satisfying "real enough to test."
- **Technical credibility**: a real scikit-learn Gradient Boosting Regression
  model (with an honestly-labelled synthetic dataset) is a genuine ML pipeline
  a viva examiner can inspect, without requiring infrastructure (databases,
  auth, deployment) that would be disproportionate for a college prototype.
  This intentionally avoids fake complexity purely to look advanced (see
  project rule: do not overengineer).
- **Usability**: a plain web form is something 3 testers (or re-engaged
  original users) can use in a few minutes without training.
  - **Feasibility**: buildable and runnable by a single student within the
  Review 1 (35% completion) timeframe.
- **Academic defensibility**: the pipeline (data → preprocessing → model →
  prediction → employee dashboard → feedback loop) mirrors the previously
  proposed architecture, so it stays consistent with prior project work and
  is easy to defend in a viva.

Runner-up considered: a pure rule-based estimator without ML. Rejected as the
sole approach because the original proposal specifically commits to Gradient
Boosting Regression; instead a **hybrid** is used — a simple rule/heuristic
baseline runs alongside the ML model so the two can be honestly compared given
limited (synthetic) training data.

## High-Level Flow (from original proposal, retained)

```
Customer / Employee
      |
Job Information
      |
Data Acquisition
      |
Pre-processing
      |
AI/ML Engine (Baseline + Gradient Boosting Regression)
      |
Predicted Service Time (range + uncertainty language)
      |
Decision Logic
      |
Employee Dashboard
      |
Customer
```

Feedback loop:
```
Actual Completion Time -> Store actual result with prediction
                        -> Evaluate model error
                        -> Periodic retraining
                        -> Gradient Boosting Regression
```

## Components Implemented in Final Product (Phase 3)

- `data/` — 800-row synthetic demonstration dataset (`synthetic_service_data.csv`), schema definition (`schema.md`), and SQLite store (`jobs.db`).
- `src/` —
  * `preprocessing.py`: Feature encodings, validation routines.
  * `model.py`: Gradient Boosting Regression and Quantile Regressors ($\alpha=0.10, 0.90$).
  * `baseline.py`: Service-type mean + linear adjustment baseline estimator.
  * `explanations.py`: Interpretive feature contribution analyzer against training medians.
  * `train.py`: Evaluation pipeline reporting MAE, RMSE, $R^2$, and empirical prediction interval coverage/width.
- `app/` —
  * `app.py`: Flask application factory (`create_app`), staff PIN authentication, customer self-service ticket tracking, bookmarkable GET URLs, dynamic cascading issue APIs, RFC-4180 CSV export, and custom error handling.
  * `job_store.py`: SQLite persistence layer managing `jobs` and `job_events` tables, safe legacy CSV migration, validated state machine, workload intelligence calculations, and active repair delay detection.
  * `templates/`: Departure-board themed Jinja templates:
    - `base.html`: Common layout, responsive topbar, and accessible footer.
    - `index.html`: Customer intake ticket form with dynamic issue cascade and steppers.
    - `result.html`: Animated departure board with prediction intervals, factor breakdown, and print styles.
    - `track.html`: Ticket lookup by 8-character reference code.
    - `track_status.html`: Live customer status tracking with 3-stage progress meter, delay alert banners, and chronological service timeline.
    - `login.html`: Staff PIN entry protection for dashboard access.
    - `dashboard.html`: Workload intelligence banner, real-time search/filters, delay pills, status transitions, and operational analytics.
    - `errors/404.html` & `errors/500.html`: Custom themed error pages.
  * `static/`:
    - `css/style.css`: Comprehensive departure-board design system, responsive breakpoints (desktop up to 960px), and print media styles.
    - `js/app.js`: Dynamic issue cascade, stepper synchronization, and client-side tracking redirection.
- `tests/` — Automated test suite with 23 end-to-end tests covering data generation, ML models, prediction intervals, validations, database transactions, timeline events, delay logic, CSV export, and authentication.
- `docs/` — Problem context, technical methodology, responsible AI, architecture, validation approaches, QA findings, and Review 1, 2, and 3 reports.

## Human-in-the-loop Responsibilities (retained from original proposal)

- Employee enters/confirms job information.
- Employee reviews the prediction before sharing it.
- Employee communicates the estimate to the customer.
- Employee updates the actual completion time (feedback loop).
- Employee can correct incorrect information.
- A human remains responsible for the final communication/decision — the
  system never presents a prediction as a guarantee.
