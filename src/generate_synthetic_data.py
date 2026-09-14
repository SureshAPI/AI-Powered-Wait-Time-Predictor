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

Usage:
    python src/generate_synthetic_data.py

Output:
    data/synthetic_service_data.csv
"""

import csv
import random
from pathlib import Path

RANDOM_SEED = 42
N_ROWS = 600

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

PHONE_BRANDS = ["Samsung", "Apple", "Xiaomi", "Realme", "OnePlus", "Vivo", "Oppo", "Other"]

PARTS_AVAILABILITY = ["Available", "Order Required", "Unknown"]


def generate_row(rng: random.Random) -> dict:
    service_type = rng.choice(list(SERVICE_TYPES.keys()))
    base_minutes, variability = SERVICE_TYPES[service_type]

    phone_brand = rng.choice(PHONE_BRANDS)
    is_warranty_case = rng.random() < 0.3
    parts_availability = rng.choices(
        PARTS_AVAILABILITY, weights=[0.65, 0.25, 0.10], k=1
    )[0]

    # Workload: how many jobs are currently ahead in queue
    current_workload = rng.randint(0, 14)
    # Technicians currently available (0 = fully booked)
    technician_availability = rng.randint(0, 4)

    duration = base_minutes

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
