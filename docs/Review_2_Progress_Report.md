# Project Better Tomorrow — Review 2 Progress Report
**AI-Powered Waiting Time Prediction for Mobile Repair and Warranty Services**  
*Continuation Track (Pathway A) — Cumulative Project Progress: 70% (Phase 2: 35%)*

---

## 1. Executive Summary
Phase 1 established the project baseline: problem framing from field visits in Gandhipuram (Coimbatore), a 600-row synthetic dataset, a baseline estimator, a Gradient Boosting Regression (GBR) pipeline, a Flask intake prototype with a "service departure board" interface, and a simulated validation study (32.2/35 marks, 92%).

Phase 2 advances the system to a complete, operational prototype bringing cumulative completion to 70%:
1. **Granular Feature Pipeline:** Separated `reported_issue` from `service_type` across 8 categories with 38 specific issues and complexity multipliers, supported by client-side dynamic cascading and server validation.
2. **Data-Driven Prediction Intervals:** Replaced the static ±20 min band with Quantile Gradient Boosting ($\alpha=0.10, 0.90$) producing an 80% empirical prediction interval.
3. **Persistent SQLite Job Store:** Implemented atomic database persistence with safe, idempotent legacy CSV migration, tagging historical records as `source='demo'` to prevent data contamination.
4. **Lifecycle State Management:** Built validated status transitions (`open` → `in_progress` → `completed` / `cancelled`) with mandatory actual-duration logging.
5. **Operational Analytics & Search:** Added live accuracy metrics (MAE, mean bias) restricted exclusively to completed operational records ($N \ge 3$ threshold) with full multi-criteria search and filtering.
6. **Interpretive Explanations:** Directional feature impact analysis comparing inputs against neutral training medians without misleading SHAP claims.
7. **Comprehensive Verification:** 17 automated tests passing (100%), verified WSGI deployment (`gunicorn app.app:app`), and zero regressions.

---

## 2. Enhanced Intake and Data Pipeline
In Phase 1, `reported_issue` was collapsed into `service_type`. Phase 2 treats them as distinct dimensions:
- **Service Categories (8):** Screen Replacement, Battery Replacement, Charging Port Repair, Software Issue, Camera Repair, Speaker/Mic Repair, Water Damage Diagnostic, Warranty Diagnostic.
- **Reported Issues (38):** Contextual symptoms (e.g., *Battery swollen*, *Stuck on boot screen*, *Camera lens cracked*). Each applies a specific complexity multiplier ($0.85\times$ to $1.20\times$) in the data generator.
- **Dataset Expansion:** Expanded to 800 rows (`data/synthetic_service_data.csv`, seed=42) for reliable quantile estimation.
- **Input Validation:** Server validation (`src/preprocessing.py`) verifies categoricals, checks issue-service compatibility, and enforces valid numerical boundaries (workload $\in [0, 30]$, technicians $\in [0, 10]$).

---

## 3. Data-Driven Prediction Intervals
Phase 1 communicated uncertainty through a static ±20-minute heuristic. Phase 2 introduces dual Quantile Gradient Boosting Regressors (`src/model.py`):
- **Lower Bound ($\alpha=0.10$):** 10th percentile duration estimator.
- **Upper Bound ($\alpha=0.90$):** 90th percentile duration estimator.
- **Evaluation on Held-Out Test Split (160 rows, seed=42):**
  - **Empirical Coverage:** **73.8%** (approximates the 80% nominal target).
  - **Average Interval Width:** **61.6 minutes** (captures high uncertainty for parts-on-order or water diagnostics).
  - **Median Interval Width:** **29.0 minutes** (captures tight bounds for routine jobs with in-stock parts).
- **Point Prediction Performance (Phase 2 dataset):**
  - Baseline (service mean + linear adjustment): MAE = 56.21 min, RMSE = 72.31 min, $R^2 = 0.010$.
  - GBR Pipeline: MAE = 23.06 min, RMSE = 38.00 min, $R^2 = 0.727$.
  *(Preserved Phase 1 Benchmark on 600 rows: Baseline MAE = 50.74 min; GBR MAE = 16.54 min, $R^2 = 0.854$.)*

---

## 4. SQLite Storage & State Lifecycle Management
`app/job_store.py` was transitioned from flat CSV appending to SQLite (`data/jobs.db`):
- **Idempotent Legacy Migration:** `_migrate_csv_if_needed()` ingests pre-existing `data/job_log.csv` entries on startup. Migrated records are stamped with `source='demo'` to permanently separate them from live records.
- **Finite-State Transitions:**
  - `open` → `in_progress`, `completed`, `cancelled`
  - `in_progress` → `completed`, `cancelled`
  - `cancelled` → `open` (reopening)
  - `completed` is an immutable terminal state.
- **Duration Audit:** Marking a job as `completed` strictly requires a positive numeric value for `actual_completion_minutes`, closing the architectural feedback loop.

---

## 5. Employee Dashboard & Live Operational Analytics
The staff dashboard (`/dashboard`) incorporates:
- **Search & Multi-Filtering:** Real-time filtering by status (`open`, `in_progress`, `completed`, `cancelled`), service type, and text search across ticket IDs, brands, and issues.
- **Strictly Separated Analytics:** Live performance metrics (AI Model MAE, Baseline MAE, Mean Prediction Error) calculate discrepancies solely on `source='live'` records where genuine completion durations were recorded.
- **Sample Safeguard:** If fewer than 3 completed live records exist, metric display is suppressed to avoid misleading claims from small samples.

---

## 6. Interpretive Prediction Explanations
Rather than misrepresenting heuristics as formal feature attribution (SHAP), `src/explanations.py` provides transparent explanations:
- Features are compared against training set neutral baselines (median workload = 7, median technicians = 2).
- **Directional Indicators:** Categorized into `increase`, `decrease`, `slight-increase`, and `neutral`.
- **Dual Presentation:** Customers receive a concise status note explaining primary drivers; employees inspect a detailed factor breakdown table on the departure board.

---

## 7. UI/UX & Deployment Integration
- **Visual Identity:** Maintained the pine-charcoal (`#10201c`), amber LED (`#f5a623`), and pale sage paper (`#f2f4ef`) departure board theme with CSS support for factor pills, validation alerts, and filter controls.
- **Client-Side Dynamics:** `app/static/js/app.js` cascades valid reported issues upon service selection and preserves state on validation warnings.
- **Production Readiness:** Structured with an application factory (`create_app`), health check endpoint (`/health`), and verified Gunicorn WSGI compatibility (`gunicorn app.app:app`).

---

## 8. Verification & Acceptance Testing
Automated test suite (`tests/test_pipeline.py`) executed via `pytest`:
- **Test Command:** `python -m pytest tests/ -v`
- **Result:** **17 passed in 12.19s (100% success rate)**
- **Coverage:** Schema verification, positive duration clamping, baseline and GBR pipelines, unseen category handling, quantile interval coverage, validation rejections, explanation directional logic, SQLite CRUD, state transition rules, idempotent migration, dashboard search/filter queries, operational analytics thresholding, HTTP route handling, and full end-to-end acceptance flow.

### Acceptance Flow Demonstration
`Customer intake submitted (/predict) → GBR generates estimate with empirical interval → Job stored in SQLite as 'open' → Employee filters dashboard and transitions status to 'in_progress' → Repair completed with actual minutes logged → Live analytics update error metrics.`

---

## 9. Remaining Limitations & Phase 3 Roadmap
- **Synthetic Training Data:** The model trains on 800 synthetic records reflecting shop dynamics; true field calibration requires collecting real longitudinal repair logs.
- **Simulated Validation:** User experience metrics remain grounded in the documented 5-persona simulated study (`docs/simulated-validation-study.md`).
- **Phase 3 Focus:** Role-based authentication (RBAC), automated model retraining triggers upon reaching $N \ge 100$ completed live jobs, and on-site field deployment.
