"""
Vitals evaluation.

Two functions, checking different things:

`eval_vitals(golden, extracted)` -- compares against a golden record. Vitals
need a different shape of eval than diagnoses: it's not about finding items
in a list, it's about five fixed fields that can each be present, null, or
wrong. Three separate questions:

1. Presence precision/recall -- did the agent report a value where one
   should exist, and only where one should exist?
2. Value accuracy -- for fields both report, are the numbers close enough?
3. Fabrication -- did the agent invent a number where the golden record
   says the value isn't actually knowable (the Eowyn / qualitative-vitals
   trap)? Kept separate from recall because it's the more dangerous
   failure: recall just means "missed something", fabrication means
   "made something up to fill a field".

`eval_vitals_plausibility(extracted)` -- no golden needed. Checks that
extracted values are physiologically plausible (e.g. catches a Fahrenheit
value left in a Celsius field, or a heart rate of 900). A value can pass
eval_vitals' checks while still being nonsense -- this catches that.

Functional style, no classes: both functions return a plain dict.
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


# --- Plausibility (no golden needed, just the extraction itself) -----------

PLAUSIBLE_RANGES = {
    "temperature_c": (30.0, 43.0),
    "heart_rate_bpm": (20, 250),
    "respiratory_rate_bpm": (4, 60),
    "spo2_percent": (50, 100),
}

BP_SYSTOLIC_RANGE = (40, 300)
BP_DIASTOLIC_RANGE = (20, 200)


def _parse_blood_pressure(bp: str):
    """'100/65' -> (100, 65). Returns None if it doesn't parse as two integers."""
    if not bp or "/" not in bp:
        return None
    try:
        systolic_str, diastolic_str = bp.split("/", 1)
        return int(systolic_str.strip()), int(diastolic_str.strip())
    except ValueError:
        return None


def eval_vitals_plausibility(extracted: dict) -> dict:
    """Checks that extracted vital sign VALUES are physiologically plausible
    -- not just present/absent (that's eval_vitals' job). Ranges are
    deliberately generous (extreme-but-survivable bounds), so this flags
    obvious extraction/unit errors, not borderline clinical values.

    Returns a plain dict:
    {
      "implausible_fields": list,          # [{"field", "value", "expected_range"}, ...]
      "unparseable_blood_pressure": bool,  # blood_pressure present but not "N/N"
    }
    """
    vitals = extracted.get("vitals", {})
    implausible = []

    for field, (low, high) in PLAUSIBLE_RANGES.items():
        value = vitals.get(field)
        if value is not None and not (low <= value <= high):
            implausible.append({"field": field, "value": value, "expected_range": [low, high]})

    unparseable_bp = False
    bp = vitals.get("blood_pressure")
    if bp is not None:
        parsed = _parse_blood_pressure(bp)
        if parsed is None:
            unparseable_bp = True
        else:
            systolic, diastolic = parsed
            if not (BP_SYSTOLIC_RANGE[0] <= systolic <= BP_SYSTOLIC_RANGE[1]):
                implausible.append({"field": "blood_pressure.systolic", "value": systolic, "expected_range": list(BP_SYSTOLIC_RANGE)})
            if not (BP_DIASTOLIC_RANGE[0] <= diastolic <= BP_DIASTOLIC_RANGE[1]):
                implausible.append({"field": "blood_pressure.diastolic", "value": diastolic, "expected_range": list(BP_DIASTOLIC_RANGE)})
            if systolic <= diastolic:
                implausible.append({"field": "blood_pressure", "value": bp, "expected_range": "systolic > diastolic"})

    return {
        "implausible_fields": implausible,
        "unparseable_blood_pressure": unparseable_bp,
    }