# Internal QA — Pre-User-Testing Pass

Performed before handing the prototype to real testers (project workflow
Phase 5). This is Claude/self-review of the running prototype, not user
testing — no real users were involved in this pass.

## Method
Probed the running Flask app with edge-case inputs (extremes of workload,
technician availability, and parts status) via automated requests, and
reviewed terminology, messaging, and data collected by the intake form.

## Findings and fixes

| # | Finding | Severity | Fix |
|---|---|---|---|
| 1 | Raw GBR prediction could be negative or near-zero for very fast/low-workload jobs (e.g. -9 minutes), which would display a nonsensical negative estimate to a customer. | High | Predictions are now clamped to a 5-minute practical floor (`MIN_PREDICTION_MINUTES` in `app/app.py`), matching the floor already used in the baseline estimator. |
| 2 | Long estimates (e.g. 229 minutes for a Water Damage + parts-on-order case) displayed as raw minutes, which is hard to read. | Medium | Added `format_minutes()` to render as hours + minutes once over 60 minutes; applied on both the customer result page and the employee dashboard. |
| 3 | Because the estimate could be negative while the ± range floor was clamped, the range shown could contradict the headline number (e.g. "~-9 minutes, typically between 5 and 11"). | High | Resolved by fix #1 — clamping happens before the range is computed, so estimate and range are always consistent. |
| 4 | Intake form was checked against `docs/responsible-ai.md`'s data-minimisation principle. | — | Confirmed: only service type, phone brand, warranty flag, parts status, workload, and technician count are collected — no name, phone number, or device identifier. No change needed. |
| 5 | Terminology check: "estimate" / "typically between" language is used consistently on both customer and dashboard views; no wording implies a guarantee. | — | No change needed. |

## Known limitations carried forward (not fixed in this pass — flagged for report/limitations section)

- The uncertainty range is a flat ±20-minute band, not a statistically derived
  prediction interval — it will read as too narrow for very long jobs (e.g.
  the ~3h49m case above) and possibly too wide for very short ones. A future
  iteration could use quantile regression for a properly calibrated interval;
  documented here rather than silently fixed so it's visible to a viva
  examiner.
- The original proposal listed a distinct "reported issue" input alongside
  service type; the current prototype folds this into `service_type` for
  simplicity. Worth explicitly noting as a scoping decision in the report.
- No authentication/multi-user handling — acceptable for a single-shop
  prototype demo, not production-ready.

## Result
Prototype is functional and internally consistent enough to proceed to real
user testing (Phase 6). See `docs/testing.md` (added at that milestone) for
the tester protocol.
