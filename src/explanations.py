"""
explanations.py

Phase 2: Generates human-readable explanations of what factors influenced
a prediction. This uses direct feature contribution analysis based on
the actual input values compared to the training data's neutral baseline
(median values), NOT a formal attribution method like SHAP.

Approach:
  - For each input feature, compare the input value against the training
    data's central tendency (median for numeric, overall mean duration for
    categorical).
  - Compute a directional indication: does this feature value tend to
    increase or decrease the predicted duration?
  - Rank factors by estimated impact magnitude.
  - Generate concise natural-language explanations.

Limitations:
  - These are approximate, interpretive explanations, not exact model
    decompositions. The GBR model's internal splits are more complex
    than what a linear comparison can capture.
  - They are designed to help customers understand "why" an estimate is
    higher or lower, not to serve as formal feature attribution.
"""

import pandas as pd


# Effect direction descriptions for categorical features, estimated
# from the synthetic data generation logic (not from model internals).
PARTS_EFFECT = {
    "Available": ("neutral", "Parts are available — no delay for ordering."),
    "Order Required": ("increase", "Parts need to be ordered, which significantly extends the wait."),
    "Unknown": ("slight increase", "Parts availability is uncertain — this may add some delay."),
}

WARRANTY_EFFECT = {
    True: ("slight increase", "Warranty cases require extra documentation time."),
    False: ("neutral", "Non-warranty job — standard processing time."),
}


def explain_prediction(inputs: dict, prediction_minutes: float,
                       training_medians: dict | None = None) -> dict:
    """Generate structured explanations for a prediction.

    Args:
        inputs: The prediction input features dict.
        prediction_minutes: The GBR point prediction in minutes.
        training_medians: Optional dict of median values from training data
            for numeric features. If None, uses reasonable defaults.

    Returns:
        A dict with:
          - 'factors': list of factor dicts, each with 'name', 'value',
            'direction' (increase/decrease/neutral), and 'explanation'.
          - 'summary': A one-sentence overall summary.
          - 'customer_summary': A shorter summary suitable for customers.
    """
    if training_medians is None:
        training_medians = {
            "current_workload": 7,       # midpoint of 0-14 range
            "technician_availability": 2,  # midpoint of 0-4 range
        }

    factors = []

    # 1. Service type + reported issue
    factors.append({
        "name": "Service type",
        "value": inputs["service_type"],
        "direction": "baseline",
        "explanation": f"Base estimate starts from typical {inputs['service_type']} duration.",
    })

    if inputs.get("reported_issue"):
        factors.append({
            "name": "Reported issue",
            "value": inputs["reported_issue"],
            "direction": "modifier",
            "explanation": f"Specific issue '{inputs['reported_issue']}' adjusts the base estimate.",
        })

    # 2. Parts availability
    parts = inputs.get("parts_availability", "Available")
    direction, explanation = PARTS_EFFECT.get(parts, ("neutral", ""))
    factors.append({
        "name": "Parts availability",
        "value": parts,
        "direction": direction,
        "explanation": explanation,
    })

    # 3. Warranty status
    is_warranty = inputs.get("is_warranty_case", False)
    direction, explanation = WARRANTY_EFFECT.get(is_warranty, ("neutral", ""))
    factors.append({
        "name": "Warranty case",
        "value": "Yes" if is_warranty else "No",
        "direction": direction,
        "explanation": explanation,
    })

    # 4. Queue workload
    workload = inputs.get("current_workload", 0)
    median_workload = training_medians.get("current_workload", 7)
    if workload > median_workload + 2:
        wl_dir = "increase"
        wl_expl = f"Queue is busier than typical ({workload} jobs ahead vs. usual ~{median_workload})."
    elif workload < median_workload - 2:
        wl_dir = "decrease"
        wl_expl = f"Queue is lighter than typical ({workload} jobs ahead vs. usual ~{median_workload})."
    else:
        wl_dir = "neutral"
        wl_expl = f"Queue load is around typical levels ({workload} jobs ahead)."
    factors.append({
        "name": "Queue workload",
        "value": f"{workload} jobs",
        "direction": wl_dir,
        "explanation": wl_expl,
    })

    # 5. Technician availability
    techs = inputs.get("technician_availability", 0)
    median_techs = training_medians.get("technician_availability", 2)
    if techs == 0:
        tech_dir = "increase"
        tech_expl = "No technician is currently free — this extends the wait."
    elif techs > median_techs + 1:
        tech_dir = "decrease"
        tech_expl = f"More technicians available than typical ({techs} vs. usual ~{median_techs})."
    elif techs < median_techs - 1:
        tech_dir = "increase"
        tech_expl = f"Fewer technicians available than typical ({techs} vs. usual ~{median_techs})."
    else:
        tech_dir = "neutral"
        tech_expl = f"Technician availability is around typical levels ({techs} available)."
    factors.append({
        "name": "Technician availability",
        "value": f"{techs} free",
        "direction": tech_dir,
        "explanation": tech_expl,
    })

    # Build summaries
    increasing = [f for f in factors if f["direction"] in ("increase",)]
    decreasing = [f for f in factors if f["direction"] in ("decrease",)]

    if increasing and not decreasing:
        summary = "Several factors are pushing the estimate higher than the base service time."
    elif decreasing and not increasing:
        summary = "Conditions are favourable — the estimate is below the typical service time."
    elif increasing and decreasing:
        summary = "Some factors increase and others decrease the estimate relative to typical conditions."
    else:
        summary = "Conditions are close to typical — the estimate reflects a standard workload scenario."

    # Shorter customer-facing summary
    if inputs.get("parts_availability") == "Order Required":
        customer_summary = "Parts need to be ordered for this job — this can extend the wait."
    elif techs == 0:
        customer_summary = "No technician is currently free — the wait may run longer than usual."
    elif workload > median_workload + 3:
        customer_summary = "The queue is busier than usual, which is adding to the estimated wait."
    elif workload < median_workload - 3 and techs > median_techs:
        customer_summary = "The shop is quieter than usual — your job should be handled sooner."
    else:
        customer_summary = "Estimate based on current shop workload and technician availability."

    return {
        "factors": factors,
        "summary": summary,
        "customer_summary": customer_summary,
    }
