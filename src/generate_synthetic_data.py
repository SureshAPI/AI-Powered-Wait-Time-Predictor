"""
generate_synthetic_data.py

Generates a SYNTHETIC / DEMONSTRATION dataset for the AI-Powered Waiting Time
Prediction prototype (Project Better Tomorrow).

IMPORTANT: This data is entirely artificial. It is shaped loosely around the
factors described in the field-visit problem context (service type, phone
brand, reported issue, current workload, technician availability, parts
availability) so that the pipeline has something realistic-looking to run
against, but it does NOT represent real historical repair records from the
Gandhipuram shop or any other shop. It must never be presented as real-world
data or real-world validation. See data/README.md and docs/methodology.md.

Phase 2 changes:
  - Added reported_issue as a separate field from service_type.
  - Each service_type has a set of plausible issue descriptions.
  - Increased dataset to 800 rows for better quantile estimation.
  - Issue complexity adds a small additional time modifier.

Usage:
    python src/generate_synthetic_data.py

Output:
    data/synthetic_service_data.csv
"""

import csv
import random
from pathlib import Path

RANDOM_SEED = 42
N_ROWS = 800

SERVICE_TYPES = {
    # service_type: (base_minutes, variability)
    "Screen Replacement": (35, 10),
    "Battery Replacement": (20, 6),
    "Charging Port Repair": (25, 8),
    "Software Issue": (15, 8),
    "Camera Repair": (30, 9),
    "Speaker/Mic Repair": (22, 7),
    "Water Damage Diagnostic": (45, 15),
    "Warranty Diagnostic": (20, 10),
}

# Phase 2: reported_issue options per service_type — separates the specific
# customer complaint from the broad service category.
REPORTED_ISSUES = {
    "Screen Replacement": [
        "Cracked screen",
        "Unresponsive touch",
        "Display flickering",
        "Dead pixels",
        "Screen discoloration",
    ],
    "Battery Replacement": [
        "Battery drains fast",
        "Phone not charging",
        "Battery swollen",
        "Overheating",
        "Shuts down randomly",
    ],
    "Charging Port Repair": [
        "Loose charging port",
        "Port not recognizing cable",
        "Slow charging",
        "Port physically damaged",
    ],
    "Software Issue": [
        "Phone keeps restarting",
        "Apps crashing",
        "Stuck on boot screen",
        "System update failed",
        "Phone running slow",
        "Virus/malware issue",
    ],
    "Camera Repair": [
        "Blurry photos",
        "Camera not opening",
        "Front camera broken",
        "Camera lens cracked",
        "Autofocus not working",
    ],
    "Speaker/Mic Repair": [
        "No sound from speaker",
        "Distorted sound",
        "Microphone not working",
        "Low call volume",
        "Earpiece crackling",
    ],
    "Water Damage Diagnostic": [
        "Phone fell in water",
        "Speaker sounds muffled after spill",
        "Screen fogging",
        "Phone not turning on after water contact",
    ],
    "Warranty Diagnostic": [
        "Manufacturing defect suspected",
        "Hardware failure within warranty",
        "Recurring issue after previous repair",
        "Dead on arrival (DOA)",
    ],
}

# Complexity multiplier — some reported issues tend to take slightly longer
# within their service category.
ISSUE_COMPLEXITY = {
    # Screen Replacement
    "Cracked screen": 1.0,
    "Unresponsive touch": 1.1,
    "Display flickering": 1.15,
    "Dead pixels": 1.05,
    "Screen discoloration": 1.1,
    # Battery Replacement
    "Battery drains fast": 1.0,
    "Phone not charging": 1.05,
    "Battery swollen": 1.15,
    "Overheating": 1.1,
    "Shuts down randomly": 1.2,
    # Charging Port Repair
    "Loose charging port": 1.0,
    "Port not recognizing cable": 1.05,
    "Slow charging": 0.95,
    "Port physically damaged": 1.15,
    # Software Issue
    "Phone keeps restarting": 1.1,
    "Apps crashing": 0.9,
    "Stuck on boot screen": 1.2,
    "System update failed": 1.1,
    "Phone running slow": 0.85,
    "Virus/malware issue": 1.15,
    # Camera Repair
    "Blurry photos": 0.95,
    "Camera not opening": 1.1,
    "Front camera broken": 1.05,
    "Camera lens cracked": 1.0,
    "Autofocus not working": 1.1,
    # Speaker/Mic Repair
    "No sound from speaker": 1.0,
    "Distorted sound": 0.95,
    "Microphone not working": 1.1,
    "Low call volume": 0.9,
    "Earpiece crackling": 1.05,
    # Water Damage Diagnostic
    "Phone fell in water": 1.0,
    "Speaker sounds muffled after spill": 0.9,
    "Screen fogging": 0.95,
    "Phone not turning on after water contact": 1.2,
    # Warranty Diagnostic
    "Manufacturing defect suspected": 1.0,
    "Hardware failure within warranty": 1.1,
    "Recurring issue after previous repair": 1.2,
    "Dead on arrival (DOA)": 0.9,
}

PHONE_BRANDS = ["Samsung", "Apple", "Xiaomi", "Realme", "OnePlus", "Vivo", "Oppo", "Other"]

PARTS_AVAILABILITY = ["Available", "Order Required", "Unknown"]


def generate_row(rng: random.Random) -> dict:
    service_type = rng.choice(list(SERVICE_TYPES.keys()))
    base_minutes, variability = SERVICE_TYPES[service_type]

    # Phase 2: pick a reported issue appropriate to the service type
    reported_issue = rng.choice(REPORTED_ISSUES[service_type])
    complexity = ISSUE_COMPLEXITY.get(reported_issue, 1.0)

    phone_brand = rng.choice(PHONE_BRANDS)
    is_warranty_case = rng.random() < 0.3
    parts_availability = rng.choices(
        PARTS_AVAILABILITY, weights=[0.65, 0.25, 0.10], k=1
    )[0]

    # Workload: how many jobs are currently ahead in queue
    current_workload = rng.randint(0, 14)
    # Technicians currently available (0 = fully booked)
    technician_availability = rng.randint(0, 4)

    duration = base_minutes * complexity

    # Workload increases wait: each queued job adds a bit of delay
    duration += current_workload * rng.uniform(1.2, 2.5)

    # More available technicians reduces wait
    duration -= technician_availability * rng.uniform(2.0, 4.0)

    # Parts not on hand adds significant delay
    if parts_availability == "Order Required":
        duration += rng.uniform(60, 240)  # minutes-equivalent delay proxy
    elif parts_availability == "Unknown":
        duration += rng.uniform(10, 30)

    # Warranty cases involve extra procedure/documentation time
    if is_warranty_case:
        duration += rng.uniform(5, 20)

    # Random noise reflecting real-world variability
    duration += rng.gauss(0, variability)

    # Duration cannot be negative or unrealistically small
    duration = max(duration, 8)

    return {
        "service_type": service_type,
        "reported_issue": reported_issue,
        "phone_brand": phone_brand,
        "is_warranty_case": is_warranty_case,
        "parts_availability": parts_availability,
        "current_workload": current_workload,
        "technician_availability": technician_availability,
        "actual_duration_minutes": round(duration, 1),
    }


def main():
    rng = random.Random(RANDOM_SEED)
    out_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_service_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [generate_row(rng) for _ in range(N_ROWS)]
    fieldnames = list(rows[0].keys())

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} SYNTHETIC rows to {out_path}")


if __name__ == "__main__":
    main()
