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

Setup and run instructions will be added here once the application exists (Milestone:
`feat: add prediction service`).

## Testing

Methodology: see [`docs/testing.md`](docs/testing.md) (added once the prototype is
testable). No user-testing results are reported until real testing has been performed.

## Responsible AI

See [`docs/responsible-ai.md`](docs/responsible-ai.md).

## Current Status

**Completed**
- Repository structure initialized.
- Problem context and continuation-track justification documented.

**Planned / Future Work**
- Synthetic dataset schema and generator.
- Baseline waiting-time estimator.
- Gradient Boosting Regression model.
- Customer intake + prediction results interface.
- Employee dashboard.
- User testing (3+ testers) and iteration.
- Final Prototype & Validation Report.

## Limitations

To be documented honestly as the prototype develops (data availability, model reliability
given limited/synthetic data, scope boundaries).

## Future Improvements

To be added following real user testing.
