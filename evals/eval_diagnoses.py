"""
Diagnoses evaluation.

Two functions, checking different things:

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

Functional style, no classes: both functions return a plain dict.
"""
import csv
from typing import Callable, Optional
import pandas as pd
from anthropic import Anthropic


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


def code_matches_diagnosis_judge_claude(diagnosis_text: str, official_description: str,
                                         model: str = "claude-sonnet-4-6") -> bool:
    """Default LLM-as-judge implementation. Requires `anthropic` + API key.
    Pass a different callable into eval_diagnoses() for testing without API calls."""
    client = Anthropic()

    prompt = (
        f'Official ICD-10 description: "{official_description}"\n'
        f'Diagnosis text written by the agent: "{diagnosis_text}"\n\n'
        "Does the diagnosis text correctly correspond to the official ICD-10 "
        "description above? Answer with exactly one word: YES or NO."
    )
    response = client.messages.create(
        model=model, max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return "YES" in response.content[0].text.upper()


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