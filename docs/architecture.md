# System Architecture

## Status
Planned architecture, prior to implementation. Will be updated to reflect what
is actually built as the prototype develops.

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

## Planned Components

- `data/` — synthetic dataset + schema documentation.
- `src/` — data loading, preprocessing, baseline estimator, GBR model,
  prediction service logic (kept as separate modules, not one large file).
- `app/` — Flask app: customer/employee intake form, prediction results view,
  employee dashboard.
- `models/` — trained model artifacts (not committed if large/binary; documented
  instead).
- `tests/` — validation tests for preprocessing and prediction logic.
- `docs/` — this architecture doc, methodology, testing methodology,
  responsible-AI notes.

## Human-in-the-loop Responsibilities (retained from original proposal)

- Employee enters/confirms job information.
- Employee reviews the prediction before sharing it.
- Employee communicates the estimate to the customer.
- Employee updates the actual completion time (feedback loop).
- Employee can correct incorrect information.
- A human remains responsible for the final communication/decision — the
  system never presents a prediction as a guarantee.
