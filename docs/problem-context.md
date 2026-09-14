# Problem Context and Continuation-Track Justification

## Continuation Track

This project continues from *C29 – Design Thinking for AI-Powered Problem
Solving*, under **Pathway A — Continuation Track**. The prior project's
Empathise and Define stages are treated as already established and are **not**
repeated. This project begins at **Ideate → Prototype → Validate → Deliver**.

## Previous Real-World Field Research (conducted by the student, not by AI)

- Location: a mobile/electronics repair shop in Gandhipuram, Coimbatore.
- A real field visit was conducted in person and real footage was recorded.
- A friend acted as an employee/staff member during the recording.
- AI was used as a thinking/analysis partner for the previous project — AI did
  not independently visit the shop, observe customers, or collect evidence.

### Observations from the field visit

These figures are **field observations/estimates from a single visit, not
independently verified or statistically validated measurements**, and are
reported here as such:

- ~30-minute observation period.
- 4 customers observed waiting during that period.
- Longest observed wait/service period: ~25 minutes.
- Estimated busy-hour volume: ~8–12 customers.

## Problem Statement

The core problem is **not** simply that customers wait too long — it is that
customers often **do not know how long** their repair/warranty service will
take. Duration varies with repair type, phone brand/model, reported issue,
current workload, technician availability, warranty procedures, and parts
availability.

This uncertainty can interfere with the customer's work, college, travel, and
other appointments.

### Existing approaches and their gap

Existing practice relies on verbal time estimates, informal queue handling,
customers physically waiting, customers calling/checking with staff, and basic
manual status communication. None of these approaches systematically combine
historical service duration, current workload, technician availability, and
service type into a reliable estimate — that is the identified gap.

## Previously Selected Solution (retained)

**AI-Powered Waiting Time Prediction for Mobile Repair and Warranty
Services** — predicting expected service/waiting time using **Gradient
Boosting Regression**, with inputs such as service type, phone brand/model,
reported issue, current job count, technician availability, historical repair
duration, and parts availability, producing a **predicted service/waiting time
communicated as an estimate/range**, not a guarantee.

### Alternatives considered previously (for reference)

Smart Digital Queue, Repair Complexity Classifier, Technician Workload
Optimiser (runner-up), AI Appointment Scheduler, Warranty Claim Assistant,
Repair Delay Detector, Customer Status Assistant, Automatic Job Information
Extraction, AI Service Dashboard. Waiting-time prediction was selected because
it offers immediate customer-facing value without requiring major operational
changes to technician allocation.

## Core Value Proposition (kept in scope)

> "Know your expected waiting/service time before you wait."

The prototype stays focused on this. It is intentionally **not** being
expanded into a full repair-shop management platform.
