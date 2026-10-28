# Project Better Tomorrow — Review 3 Progress Report
**AI-Powered Waiting Time Prediction for Mobile Repair and Warranty Services**  
*Continuation Track (Pathway A) — Cumulative Progress: 100% (Phase 3: Final 30%)*

---

## 1. Executive Summary
Project Better Tomorrow addresses a verified customer problem discovered during field visits in Gandhipuram, Coimbatore: customers at electronics and warranty repair centers experience deep uncertainty regarding repair turnaround times.

- **Phase 1 (35% completion, 32.2/35 marks):** Built baseline estimator, Gradient Boosting Regression (GBR) pipeline, synthetic dataset (600 rows), departure board UI prototype, and simulated validation study.
- **Phase 2 (70% cumulative completion, 32.2/35 marks):** Decoupled service types from 38 reported issues, introduced Quantile GBR prediction intervals ($\alpha=0.10, 0.90$), implemented SQLite persistence (`jobs` table) with safe legacy CSV migration, established validated lifecycle transitions, operational analytics for live completed records, and transparent feature explanations.
- **Phase 3 (100% completion — Final Deliverable):** Transforms the prototype into a complete, secure, production-ready product:
  1. **Customer Self-Service Tracking:** Public ticket lookup (`/track`) and real-time status portal (`/track/<job_id>`) with a 3-stage visual progress meter.
  2. **Audit-Ready Service Timeline:** Built an event sourcing table (`job_events`) logging timestamps and descriptions for every state transition.
  3. **Queue Workload Intelligence:** Upgraded the employee dashboard with queue depth, backlog wait duration, overdue warnings, and high-variance job flags.
  4. **Proactive Delay & Uncertainty Alerts:** Runtime delay monitoring alerts staff and customers when active repair time exceeds the predicted estimate.
  5. **Staff Access Control:** Workshop PIN authentication (`/dashboard/login`, session-backed) protecting administrative routes while keeping customer tracking public.
  6. **Export & Print Capabilities:** One-click print-optimized estimate receipts (`@media print`) and RFC-4180 CSV export (`/dashboard/export`) preserving active query filters.
  7. **Comprehensive Verification:** Expanded automated test suite to 23 tests (100% pass rate) and verified Gunicorn WSGI deployment.

---

## 2. Customer Tracking & Service Timeline
In earlier phases, customers could only view their estimate right after intake. Phase 3 provides continuous visibility without exposing internal shop data:
- **Reference-Based Lookup (`/track`):** Customers enter their 8-character ticket reference code (e.g., `4c37167e`) generated upon intake.
- **Live Status Tracking (`/track/<job_id>`):** Displays real-time status (`QUEUED`, `IN PROGRESS`, `COMPLETED`, `CANCELLED`) with animated indicators and a 3-stage progress meter (*Intake Received* $\to$ *Bench Service* $\to$ *Ready for Pickup*).
- **Service Timeline (`job_events` table):** A dedicated SQLite table records all lifecycle milestones (`created`, `in_progress`, `completed`, `cancelled`, `reopened`) with UTC timestamps and annotations.
- **Fallback Resilience:** For historical demo records, the event engine dynamically constructs timeline milestones from creation data.

---

## 3. Queue Workload Intelligence
The staff dashboard (`/dashboard`) was augmented with operational metrics for workshop coordination:
- **Active In-Shop Queue:** Total count of active repairs on site (`open` + `in_progress`).
- **Total Backlog Wait Duration:** Cumulative predicted minutes remaining across pending jobs, formatted into hours and minutes.
- **Delay / Overdue Warning Counter:** Highlights active jobs where elapsed time from registration has exceeded the AI-predicted completion estimate.
- **High Variance Indicator:** Tracks jobs with wide empirical prediction intervals ($\ge 45$ min) or pending component orders.
- **Visual Card Highlights:** Overdue jobs are visually marked with a crimson border and a `DELAYED +X min` badge for immediate technician prioritization.

---

## 4. Prediction Uncertainty & Delay Detection
Ethical AI communication requires transparent uncertainty representation:
- **High Uncertainty Alert:** Jobs requiring component ordering (`parts_availability = 'Order Required'`) or exhibiting broad quantile spreads ($\text{Upper}_{90} - \text{Lower}_{10} \ge 45$ min) trigger an alert explaining that component sourcing creates wider turnaround variability.
- **Elapsed-Time Delay Monitoring (`check_job_delay`):** An analytical routine calculates elapsed minutes against predicted minutes ($t_{\text{elapsed}} > t_{\text{predicted}}$).
- **Customer Notification:** The customer tracking board proactively alerts customers if diagnostic complications extend repair time.

---

## 5. Staff Access Control & System Hardening
- **Staff PIN Authentication (`/dashboard/login`):** Dashboard management, status updates, completion logging, and data exports are protected by session-based PIN authentication (`STAFF_PIN`, default `1234`).
- **Scoped Public Access:** Customer tracking (`/track/<job_id>`) remains open via reference code without requiring login, preventing customer data enumeration while keeping tracking effortless.
- **Safe Session Management:** Session tokens persist across browser sessions with a clean logout route (`/dashboard/logout`).
- **Custom Error Handling:** Branded departure-board error handlers for 404 and 500 guide users back to safety without exposing stack traces.

---

## 6. Practical Production Utilities
- **Printable Estimate Summary:** Formatted with a clean print button on results and tracking screens. A comprehensive `@media print` CSS stylesheet strips navigation bars, action buttons, and background colors to output clean monochrome physical receipts.
- **RFC-4180 CSV Export (`/dashboard/export`):** Enables shop owners to export filtered datasets directly into spreadsheet applications, preserving active query filters (`status`, `search`, `service_type`, `source`).

---

## 7. Verification & Automated Testing
The automated test suite (`tests/test_pipeline.py`) was expanded with 6 new end-to-end Phase 3 test cases:
- **Test Command:** `python -m pytest tests/ -v`
- **Results:** **23 passed in 4.38s (100% pass rate)**
- **New Test Coverage:**
  1. `test_job_events_logging_and_timeline`: Validates event creation across status transitions with timestamp accuracy.
  2. `test_workload_intelligence_and_delay_detection`: Verifies active backlog calculations and simulated overdue detection ($t_{\text{elapsed}} > 120$ min).
  3. `test_customer_ticket_tracking_workflow`: Tests search forms, query redirects, 404 responses, and live tracking pages.
  4. `test_csv_export_endpoint`: Validates RFC-4180 CSV generation and attachment headers.
  5. `test_staff_authentication_and_access_control`: Tests PIN entry, invalid PIN rejection, session auth, and protected route redirection.
  6. `test_custom_404_error_page`: Tests custom branded error pages.
- **WSGI Verification:** `python -c "from app.app import app"` verified clean startup for Gunicorn deployment (`gunicorn app.app:app`).

---

## 8. Academic Honesty & Limitations
- **Synthetic Training Data:** The GBR model remains trained on 800 synthetic records reflecting shop dynamics; genuine real-world accuracy claims require long-term shop logging.
- **Simulated Validation:** User feedback metrics remain documented under the simulated validation study (`docs/simulated-validation-study.md`).
- **Single-Workshop Architecture:** SQLite was chosen for atomic, reliable single-workshop operations without the overhead of cloud databases.

---

## 9. Conclusion
Phase 3 completes Project Better Tomorrow, bringing the project to 100% completion. The system successfully combines machine learning, quantile uncertainty quantification, responsive UI design, customer tracking, workload intelligence, and access controls into a defensible college engineering project.
