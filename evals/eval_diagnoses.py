"""
Diagnoses evaluation.

Several functions, checking different things:

`eval_diagnoses(golden, extracted, icd10_catalog)` -- compares against a
golden record. Two separate questions, deliberately kept separate because
each catches a different failure mode:

1. Precision / Recall on ICD-10 codes (extracted vs. golden) -- did the
   agent find the right diagnoses, and only the right ones?
2. Are the codes it used real? (looked up in the ICD-10 catalog, not
   invented from memory)
3. Does the diagnosis text actually correspond to what that code means?
   (an agent can use a *valid* code that just doesn't match its own text)

`eval_diagnosis_list_limits(extracted)` -- no golden needed. Checks
structural limits diagnoses_prompt.txt mandates directly (max 3 entries per
list, assessment can't be empty).

`eval_diagnosis_code_format(extracted)` -- no golden needed. Checks every
code has the standard ICD-10-CM shape, independent of whether it's actually
in the catalog (a malformed code fails fast, before even bothering to look
it up).

Functional style, no classes: every function returns a plain dict.
"""
import csv
import logging
import re
from typing import Callable, Optional
import pandas as pd

logger = logging.getLogger("eval_diagnoses")

# Score configs for log_diagnoses_scores()/log_diagnosis_list_limits_scores().
# Pass through judge_client.ensure_score_configs(client, SCORE_CONFIGS) once
# per process to get the {name: config_id} map those functions expect.
SCORE_CONFIGS = [
    {
        "name": "diagnoses_precision",
        "data_type": "NUMERIC",
        "min_value": 0,
        "max_value": 1,
        "description": "Precision of extracted ICD-10 codes against the golden record.",
    },
    {
        "name": "diagnoses_recall",
        "data_type": "NUMERIC",
        "min_value": 0,
        "max_value": 1,
        "description": "Recall of extracted ICD-10 codes against the golden record.",
    },
    {
        "name": "diagnoses_has_invalid_codes",
        "data_type": "BOOLEAN",
        "description": "Whether any extracted ICD-10 code was not found in the ICD-10 catalog.",
    },
    {
        "name": "diagnoses_has_code_mismatches",
        "data_type": "BOOLEAN",
        "description": "Whether any extracted diagnosis text didn't match its ICD-10 code's official description.",
    },
    {
        "name": "diagnoses_assessment_too_long",
        "data_type": "BOOLEAN",
        "description": "Whether the assessment list exceeded MAX_DIAGNOSES entries.",
    },
    {
        "name": "diagnoses_differential_too_long",
        "data_type": "BOOLEAN",
        "description": "Whether the differential diagnoses list exceeded MAX_DIAGNOSES entries.",
    },
    {
        "name": "diagnoses_assessment_empty",
        "data_type": "BOOLEAN",
        "description": "Whether the assessment list had zero entries (violates 'MUST have at least one').",
    },
    {
        "name": "diagnoses_has_malformed_codes",
        "data_type": "BOOLEAN",
        "description": "Whether any extracted ICD-10 code doesn't match the standard ICD-10-CM shape.",
    },
]


def load_icd10_catalog(path: str) -> dict[str, str]:
    """Loads a CSV with columns: code, description -> {code: description}"""
    catalog = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            catalog[row["code"]] = row["description"]
    return catalog


def load_icd10_catalog_parquet(
    path: str, code_col: str = "ICD10_Code", description_col: str = "Description"
) -> dict[str, str]:
    """Loads the catalog from a .parquet file instead of CSV -- e.g. the
    ICD10_DB.parquet in this repo's data/ folder. Adjust code_col /
    description_col if your parquet uses different column names (check
    with `pandas.read_parquet(path).columns` first)."""
    df = pd.read_parquet(path)
    return dict(zip(df[code_col], df[description_col]))


def _all_diagnosis_items(note: dict) -> list[dict]:
    """Assessment + differential entries, uniformly shaped."""
    items = []
    for item in note.get("assessment", []):
        items.append({"diagnosis": item["diagnosis"], "icd10_code": item.get("icd10_code")})
    for item in note.get("differential_diagnoses", []):
        items.append({"diagnosis": item["diagnosis"], "icd10_code": item.get("icd10_code")})
    return items


def _codes(note: dict) -> set[str]:
    return {i["icd10_code"] for i in _all_diagnosis_items(note) if i["icd10_code"]}


def eval_diagnoses(
    golden: dict,
    extracted: dict,
    icd10_catalog: dict[str, str],
    code_matches_diagnosis_judge: Optional[Callable[[str, str], bool]] = None,
) -> dict:
    """Returns a plain dict:
    {
      "precision": float,
      "recall": float,
      "false_positive_codes": set,
      "false_negative_codes": set,
      "invalid_codes": set,               # codes not found in the ICD-10 catalog at all
      "code_diagnosis_mismatches": list,   # code is valid, but the diagnosis text doesn't match it
    }
    """
    golden_codes = _codes(golden)
    extracted_codes = _codes(extracted)

    true_positives = golden_codes & extracted_codes
    false_positives = extracted_codes - golden_codes
    false_negatives = golden_codes - extracted_codes

    precision = len(true_positives) / len(extracted_codes) if extracted_codes else 1.0
    recall = len(true_positives) / len(golden_codes) if golden_codes else 1.0

    invalid_codes = {c for c in extracted_codes if c not in icd10_catalog}

    mismatches = []
    if code_matches_diagnosis_judge:
        for item in _all_diagnosis_items(extracted):
            code = item["icd10_code"]
            if not code or code not in icd10_catalog:
                continue  # already flagged as invalid above, don't double-report
            official_description = icd10_catalog[code]
            if not code_matches_diagnosis_judge(item["diagnosis"], official_description):
                mismatches.append({
                    "code": code,
                    "diagnosis_text": item["diagnosis"],
                    "official_description": official_description,
                })

    return {
        "precision": precision,
        "recall": recall,
        "false_positive_codes": false_positives,
        "false_negative_codes": false_negatives,
        "invalid_codes": invalid_codes,
        "code_diagnosis_mismatches": mismatches,
    }


def log_diagnoses_scores(
    client,
    trace_id: str,
    result: dict,
    observation_id: Optional[str] = None,
    config_ids: Optional[dict[str, str]] = None,
) -> None:
    """Attach eval_diagnoses()'s metrics to a Langfuse trace (optionally scoped
    to one observation) as scores. `config_ids` is the {name: config_id} map
    from judge_client.ensure_score_configs(client, SCORE_CONFIGS)."""
    config_ids = config_ids or {}
    scores = [
        ("diagnoses_precision", result["precision"], "NUMERIC"),
        ("diagnoses_recall", result["recall"], "NUMERIC"),
        ("diagnoses_has_invalid_codes", bool(result["invalid_codes"]), "BOOLEAN"),
        ("diagnoses_has_code_mismatches", bool(result["code_diagnosis_mismatches"]), "BOOLEAN"),
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
        "trace %s: diagnoses precision=%.2f recall=%.2f", trace_id, result["precision"], result["recall"]
    )
    if result["invalid_codes"]:
        logger.warning("trace %s: invalid ICD-10 codes: %s", trace_id, sorted(result["invalid_codes"]))
    if result["code_diagnosis_mismatches"]:
        logger.warning(
            "trace %s: %d code/diagnosis text mismatch(es)", trace_id, len(result["code_diagnosis_mismatches"])
        )


# --- Structural limits (no golden needed, just the extraction itself) -------

MAX_DIAGNOSES = 3


def eval_diagnosis_list_limits(extracted: dict) -> dict:
    """Checks structural limits that diagnoses_prompt.txt /
    diagnoses_prompt_with_tool.txt mandate, independent of any golden
    comparison:

    - differential_diagnoses: at most 3 entries ("report only the 3 most relevant")
    - assessment: at most 3 entries (same rule) AND at least 1 entry
      ("Every encounter MUST have at least one primary diagnosis")

    Returns a plain dict:
    {
      "assessment_count": int,
      "differential_count": int,
      "assessment_too_long": bool,    # more than 3 entries
      "differential_too_long": bool,  # more than 3 entries
      "assessment_empty": bool,       # zero entries -- violates "MUST have at least one"
    }
    """
    assessment = extracted.get("assessment", [])
    differentials = extracted.get("differential_diagnoses", [])

    return {
        "assessment_count": len(assessment),
        "differential_count": len(differentials),
        "assessment_too_long": len(assessment) > MAX_DIAGNOSES,
        "differential_too_long": len(differentials) > MAX_DIAGNOSES,
        "assessment_empty": len(assessment) == 0,
    }


def log_diagnosis_list_limits_scores(
    client,
    trace_id: str,
    result: dict,
    observation_id: Optional[str] = None,
    config_ids: Optional[dict[str, str]] = None,
) -> None:
    """Attach eval_diagnosis_list_limits()'s metrics to a Langfuse trace as
    BOOLEAN scores. `config_ids` is the {name: config_id} map from
    judge_client.ensure_score_configs(client, SCORE_CONFIGS)."""
    config_ids = config_ids or {}
    scores = [
        ("diagnoses_assessment_too_long", result["assessment_too_long"]),
        ("diagnoses_differential_too_long", result["differential_too_long"]),
        ("diagnoses_assessment_empty", result["assessment_empty"]),
    ]
    for name, value in scores:
        client.create_score(
            name=name,
            value=int(value),
            trace_id=trace_id,
            observation_id=observation_id,
            data_type="BOOLEAN",
            config_id=config_ids.get(name),
        )
    if result["assessment_too_long"] or result["differential_too_long"] or result["assessment_empty"]:
        logger.warning("trace %s: diagnosis list limit violation(s): %s", trace_id, {
            k: v for k, v in result.items() if k.endswith(("_too_long", "_empty")) and v
        })


# --- ICD-10 code format (no golden needed, just the extraction itself) -----

# Standard ICD-10-CM shape: one letter, two digits, then an optional decimal
# point followed by up to four more alphanumeric characters (e.g. "J18.9",
# "S41.032A", "T14.8XXA"). Independent of catalog membership -- a code can
# match this shape and still not exist (that's `invalid_codes` in
# eval_diagnoses), or fail this shape outright (missing the dot, lowercase,
# extra segments) which is a cheaper, faster signal to catch first.
ICD10_CODE_PATTERN = re.compile(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$")


def eval_diagnosis_code_format(extracted: dict) -> dict:
    """Checks every extracted ICD-10 code matches the standard ICD-10-CM shape.

    Returns a plain dict:
    {"malformed_codes": set}  # codes present but not matching ICD10_CODE_PATTERN
    """
    codes = _codes(extracted)
    malformed = {c for c in codes if not ICD10_CODE_PATTERN.match(c)}
    return {"malformed_codes": malformed}


def log_diagnosis_code_format_scores(
    client,
    trace_id: str,
    result: dict,
    observation_id: Optional[str] = None,
    config_ids: Optional[dict[str, str]] = None,
) -> None:
    """Attach eval_diagnosis_code_format()'s result to a Langfuse trace as a
    single BOOLEAN score. `config_ids` is the {name: config_id} map from
    judge_client.ensure_score_configs(client, SCORE_CONFIGS)."""
    config_ids = config_ids or {}
    client.create_score(
        name="diagnoses_has_malformed_codes",
        value=int(bool(result["malformed_codes"])),
        trace_id=trace_id,
        observation_id=observation_id,
        data_type="BOOLEAN",
        config_id=config_ids.get("diagnoses_has_malformed_codes"),
    )
    if result["malformed_codes"]:
        logger.warning("trace %s: malformed ICD-10 codes: %s", trace_id, sorted(result["malformed_codes"]))


def run_self_tests() -> list[str]:
    """Fabricated single-error scenarios built directly from
    data/golden/golden_encounter_{riv001,hou002}.json -- no invented patient,
    no other golden file. Each scenario starts from a *perfect* extraction
    (an exact copy of golden's own differentials/assessment) and changes
    exactly one thing, then asserts eval_diagnoses/eval_diagnosis_list_limits
    actually catch it. Returns the list of failed scenario labels (empty if
    everything passed). Run via `uv run python evals/eval_diagnoses.py`.
    """
    import copy
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    golden = {
        name: json.loads((repo_root / f"data/golden/golden_encounter_{name}.json").read_text(encoding="utf-8"))
        for name in ("riv001", "hou002")
    }
    catalog = load_icd10_catalog_parquet(str(repo_root / "data/ICD10_DB.parquet"))

    def perfect_extraction(g: dict) -> dict:
        return {
            "differential_diagnoses": copy.deepcopy(g["differential_diagnoses"]),
            "assessment": copy.deepcopy(g["assessment"]),
        }

    def codes(g: dict) -> set:
        return {i["icd10_code"] for i in g["differential_diagnoses"] + g["assessment"]}

    def a_real_code_not_in(g: dict) -> str:
        used = codes(g)
        return next(c for c in catalog if c not in used)

    failed = []

    def check(label: str, condition: bool, detail="") -> None:
        if condition:
            print(f"  PASS  {label}")
        else:
            failed.append(label)
            print(f"  FAIL  {label}  {detail}")

    print("=== Diagnoses (eval_diagnoses / eval_diagnosis_list_limits) ===")

    g = golden["riv001"]

    extracted = perfect_extraction(g)
    extracted["assessment"][0]["icd10_code"] = "ZZ00.00"
    result = eval_diagnoses(g, extracted, catalog)
    check("invalid_icd_code -> flagged in invalid_codes", "ZZ00.00" in result["invalid_codes"], result)

    g2 = golden["hou002"]
    extracted = perfect_extraction(g2)
    extra_code = a_real_code_not_in(g2)
    extracted["differential_diagnoses"].append({"icd10_code": extra_code, "diagnosis": "Unrelated made-up diagnosis"})
    result = eval_diagnoses(g2, extracted, catalog)
    check(
        "false_positive_diagnosis -> in false_positive_codes, precision < 1",
        extra_code in result["false_positive_codes"] and result["precision"] < 1.0,
        result,
    )

    extracted = perfect_extraction(g)
    dropped = extracted["differential_diagnoses"].pop(0)["icd10_code"]
    result = eval_diagnoses(g, extracted, catalog)
    check(
        "false_negative_diagnosis -> in false_negative_codes, recall < 1",
        dropped in result["false_negative_codes"] and result["recall"] < 1.0,
        result,
    )

    extracted = perfect_extraction(g)
    padding_code = a_real_code_not_in(g)
    extracted["assessment"] += [{"icd10_code": padding_code, "diagnosis": "padding"}] * 3
    result = eval_diagnosis_list_limits(extracted)
    check("assessment_too_long -> flagged", result["assessment_too_long"] is True, result)

    extracted = perfect_extraction(g)
    padding_code = a_real_code_not_in(g)
    extracted["differential_diagnoses"] += [{"icd10_code": padding_code, "diagnosis": "padding"}] * 3
    result = eval_diagnosis_list_limits(extracted)
    check("differential_too_long -> flagged", result["differential_too_long"] is True, result)

    extracted = perfect_extraction(g)
    extracted["assessment"] = []
    result = eval_diagnosis_list_limits(extracted)
    check("assessment_empty -> flagged", result["assessment_empty"] is True, result)

    extracted = perfect_extraction(g)
    result = eval_diagnosis_code_format(extracted)
    check("perfect_extraction -> no malformed_codes", not result["malformed_codes"], result)

    extracted = perfect_extraction(g)
    extracted["assessment"][0]["icd10_code"] = "S41032A"  # missing the decimal point
    result = eval_diagnosis_code_format(extracted)
    check("malformed_code -> flagged in malformed_codes", "S41032A" in result["malformed_codes"], result)

    print(f"\n{len(failed)} failed" if failed else "\nall passed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(1 if run_self_tests() else 0)