"""
Vitals evaluation.

Vitals need a different shape of eval than diagnoses: it's not about finding
items in a list, it's about five fixed fields that can each be present,
null, or wrong. Three separate questions:

1. Presence precision/recall -- did the agent report a value where one
   should exist, and only where one should exist?
2. Value accuracy -- for fields both report, are the numbers close enough?
3. Fabrication -- did the agent invent a number where the golden record
   says the value isn't actually knowable (the Eowyn / qualitative-vitals
   trap)? Kept separate from recall because it's the more dangerous
   failure: recall just means "missed something", fabrication means
   "made something up to fill a field".

Functional style, no classes: eval_vitals() returns a plain dict.
"""

VITAL_FIELDS = [
    "temperature_c", "heart_rate_bpm", "respiratory_rate_bpm",
    "blood_pressure", "spo2_percent",
]

# Absolute tolerance per field before a present-in-both value counts as a mismatch
TOLERANCE = {
    "temperature_c": 0.2,
    "heart_rate_bpm": 3,
    "respiratory_rate_bpm": 2,
    "spo2_percent": 2,
}


def eval_vitals(golden: dict, extracted: dict) -> dict:
    """Returns a plain dict:
    {
      "presence_precision": float,
      "presence_recall": float,
      "value_mismatches": list,
      "fabricated_fields": list,
      "missing_fields": list,
    }
    """
    g_vitals = golden["vitals"]
    e_vitals = extracted["vitals"]

    reported_by_agent = {f for f in VITAL_FIELDS if e_vitals.get(f) is not None}
    should_be_reported = {f for f in VITAL_FIELDS if g_vitals.get(f) is not None}

    true_positive_fields = reported_by_agent & should_be_reported
    presence_precision = (
        len(true_positive_fields) / len(reported_by_agent) if reported_by_agent else 1.0
    )
    presence_recall = (
        len(true_positive_fields) / len(should_be_reported) if should_be_reported else 1.0
    )

    fabricated = [
        f for f in VITAL_FIELDS
        if g_vitals.get(f) is None and e_vitals.get(f) is not None
    ]
    missing = [
        f for f in VITAL_FIELDS
        if g_vitals.get(f) is not None and e_vitals.get(f) is None
    ]

    mismatches = []
    for f in true_positive_fields:
        g_val, e_val = g_vitals[f], e_vitals[f]
        if f == "blood_pressure":
            if g_val != e_val:
                mismatches.append({"field": f, "golden": g_val, "extracted": e_val})
        else:
            tol = TOLERANCE.get(f, 0)
            if abs(g_val - e_val) > tol:
                mismatches.append({"field": f, "golden": g_val, "extracted": e_val})

    return {
        "presence_precision": presence_precision,
        "presence_recall": presence_recall,
        "value_mismatches": mismatches,
        "fabricated_fields": fabricated,
        "missing_fields": missing,
    }
