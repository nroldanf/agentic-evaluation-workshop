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

import logging
from typing import Optional

logger = logging.getLogger("eval_vitals")

# Score configs for log_vitals_scores()/log_vitals_plausibility_scores().
# Pass through judge_client.ensure_score_configs(client, SCORE_CONFIGS) once
# per process to get the {name: config_id} map those functions expect.
SCORE_CONFIGS = [
    {
        "name": "vitals_presence_precision",
        "data_type": "NUMERIC",
        "min_value": 0,
        "max_value": 1,
        "description": "Precision of reported vitals fields against the golden record.",
    },
    {
        "name": "vitals_presence_recall",
        "data_type": "NUMERIC",
        "min_value": 0,
        "max_value": 1,
        "description": "Recall of reported vitals fields against the golden record.",
    },
    {
        "name": "vitals_has_value_mismatch",
        "data_type": "BOOLEAN",
        "description": "Whether any vitals field present in both golden and extracted differed beyond tolerance.",
    },
    {
        "name": "vitals_has_fabricated_fields",
        "data_type": "BOOLEAN",
        "description": "Whether the agent reported a vitals value the golden record says isn't knowable.",
    },
    {
        "name": "vitals_plausible",
        "data_type": "BOOLEAN",
        "description": "Whether all extracted vitals values fall within physiologically plausible ranges.",
    },
]

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


def log_vitals_scores(
    client,
    trace_id: str,
    result: dict,
    observation_id: Optional[str] = None,
    config_ids: Optional[dict[str, str]] = None,
) -> None:
    """Attach eval_vitals()'s metrics to a Langfuse trace (optionally scoped to
    one observation) as scores. `config_ids` is the {name: config_id} map from
    judge_client.ensure_score_configs(client, SCORE_CONFIGS)."""
    config_ids = config_ids or {}
    scores = [
        ("vitals_presence_precision", result["presence_precision"], "NUMERIC"),
        ("vitals_presence_recall", result["presence_recall"], "NUMERIC"),
        ("vitals_has_value_mismatch", bool(result["value_mismatches"]), "BOOLEAN"),
        ("vitals_has_fabricated_fields", bool(result["fabricated_fields"]), "BOOLEAN"),
    ]
    for name, value, data_type in scores:
        # Boolean scores must be ingested as 0/1, not Python bool (see
        # Langfuse docs: "Boolean scores must be provided as a float").
        client.create_score(
            name=name,
            value=int(value) if data_type == "BOOLEAN" else value,
            trace_id=trace_id,
            observation_id=observation_id,
            data_type=data_type,
            config_id=config_ids.get(name),
        )
    logger.info(
        "trace %s: vitals presence_precision=%.2f presence_recall=%.2f",
        trace_id, result["presence_precision"], result["presence_recall"],
    )
    if result["fabricated_fields"]:
        logger.warning("trace %s: fabricated vitals fields: %s", trace_id, result["fabricated_fields"])
    if result["value_mismatches"]:
        logger.warning("trace %s: %d vitals value mismatch(es)", trace_id, len(result["value_mismatches"]))


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


def log_vitals_plausibility_scores(
    client,
    trace_id: str,
    result: dict,
    observation_id: Optional[str] = None,
    config_ids: Optional[dict[str, str]] = None,
) -> None:
    """Attach eval_vitals_plausibility()'s result to a Langfuse trace as a
    single BOOLEAN score. `config_ids` is the {name: config_id} map from
    judge_client.ensure_score_configs(client, SCORE_CONFIGS)."""
    config_ids = config_ids or {}
    plausible = not result["implausible_fields"] and not result["unparseable_blood_pressure"]
    client.create_score(
        name="vitals_plausible",
        value=int(plausible),
        trace_id=trace_id,
        observation_id=observation_id,
        data_type="BOOLEAN",
        config_id=config_ids.get("vitals_plausible"),
    )
    if not plausible:
        logger.warning(
            "trace %s: implausible vitals -- implausible_fields=%s unparseable_blood_pressure=%s",
            trace_id, result["implausible_fields"], result["unparseable_blood_pressure"],
        )


def run_self_tests() -> list[str]:
    """Fabricated single-error scenarios built directly from
    data/golden/golden_encounter_{riv001,hou002}.json -- no invented patient,
    no other golden file. Each scenario starts from a *perfect* extraction
    (an exact copy of golden's own vitals) and changes exactly one thing,
    then asserts eval_vitals/eval_vitals_plausibility actually catch it.
    Returns the list of failed scenario labels (empty if everything passed).
    Run via `uv run python evals/eval_vitals.py`.
    """
    import copy
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    golden = {
        name: json.loads((repo_root / f"data/golden/golden_encounter_{name}.json").read_text(encoding="utf-8"))
        for name in ("riv001", "hou002")
    }

    def perfect_extraction(g: dict) -> dict:
        return {"vitals": copy.deepcopy(g["vitals"])}

    failed = []

    def check(label: str, condition: bool, detail="") -> None:
        if condition:
            print(f"  PASS  {label}")
        else:
            failed.append(label)
            print(f"  FAIL  {label}  {detail}")

    print("=== Vitals (eval_vitals / eval_vitals_plausibility) ===")

    g = golden["riv001"]

    extracted = perfect_extraction(g)
    extracted["vitals"]["temperature_c"] = 100
    result = eval_vitals_plausibility(extracted)
    check(
        "implausible_temperature -> flagged",
        any(f["field"] == "temperature_c" for f in result["implausible_fields"]),
        result,
    )

    extracted = perfect_extraction(g)
    extracted["vitals"]["heart_rate_bpm"] = 900
    result = eval_vitals_plausibility(extracted)
    check(
        "implausible_heart_rate -> flagged",
        any(f["field"] == "heart_rate_bpm" for f in result["implausible_fields"]),
        result,
    )

    # RIV-001's golden reports all 5 fields -- HOU-002's golden leaves
    # respiratory_rate_bpm null, so that's the one that can test fabrication.
    g2 = golden["hou002"]
    assert g2["vitals"]["respiratory_rate_bpm"] is None
    extracted = perfect_extraction(g2)
    extracted["vitals"]["respiratory_rate_bpm"] = 16
    result = eval_vitals(g2, extracted)
    check("fabricated_field -> in fabricated_fields", "respiratory_rate_bpm" in result["fabricated_fields"], result)

    extracted = perfect_extraction(g)
    extracted["vitals"]["temperature_c"] = None
    result = eval_vitals(g, extracted)
    check(
        "missing_field -> in missing_fields, recall < 1",
        "temperature_c" in result["missing_fields"] and result["presence_recall"] < 1.0,
        result,
    )

    # Value mismatch that's still physiologically plausible -- distinct
    # signal from implausibility (both fields present, just too far apart
    # per TOLERANCE).
    extracted = perfect_extraction(g)
    extracted["vitals"]["heart_rate_bpm"] = 115  # golden=102, tolerance=3, still well within 20-250
    vitals_result = eval_vitals(g, extracted)
    plausibility_result = eval_vitals_plausibility(extracted)
    check(
        "value_mismatch_within_plausible_range -> mismatch flagged, NOT implausible",
        any(m["field"] == "heart_rate_bpm" for m in vitals_result["value_mismatches"])
        and not any(f["field"] == "heart_rate_bpm" for f in plausibility_result["implausible_fields"]),
        (vitals_result["value_mismatches"], plausibility_result["implausible_fields"]),
    )

    extracted = perfect_extraction(g)
    extracted["vitals"]["blood_pressure"] = "unknown"
    result = eval_vitals_plausibility(extracted)
    check("unparseable_blood_pressure -> flagged", result["unparseable_blood_pressure"] is True, result)

    extracted = perfect_extraction(g)
    extracted["vitals"]["blood_pressure"] = "60/120"
    result = eval_vitals_plausibility(extracted)
    check(
        "blood_pressure_systolic_not_greater -> flagged",
        any(f["field"] == "blood_pressure" for f in result["implausible_fields"]),
        result,
    )

    print(f"\n{len(failed)} failed" if failed else "\nall passed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(1 if run_self_tests() else 0)