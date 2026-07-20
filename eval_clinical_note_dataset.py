"""Regression-test HPI + Physical Exam extraction on one shared golden dataset.

Companion to `eval_hpi_judge.py` (which scores whichever trace was most
recently produced — the online/production-monitoring lane). This script is
the offline/regression lane: one Langfuse Dataset (`data/encounter_1.txt`,
`data/encounter_2.txt`), one task that runs both the `hpi` and
`physical_exam` nodes per item, and all 8 evaluators attached to the same
Experiment run — so a single DatasetRun in the Langfuse UI shows every
metric for both sections together, rather than two separate runs to
cross-reference.

Metrics, one LLM call per section (see `prompts/hpi_judge_prompt.txt` /
`prompts/physical_exam_judge_prompt.txt`):
  - hpi_accuracy / hpi_completeness / hpi_tone
  - pe_accuracy / pe_completeness / pe_tone
  - pe_precision / pe_recall (code, deterministic — system-set match against
    the golden physical exam; see `_is_filler`'s docstring for its one known
    limitation)

Re-running this script (e.g. after switching `OLLAMA_MODEL` or editing a
prompt) creates a new, separately named DatasetRun for side-by-side
comparison in the UI.
"""

import argparse
import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Evaluation, get_client

from agent import (
    OLLAMA_MODEL,
    build_agent,
    extraction_model,
    get_callbacks,
    hpi_prompt,
    physical_exam_prompt,
    run_agent,
)
from eval_hpi_judge import JUDGE_PROMPT_PATH as HPI_JUDGE_PROMPT_PATH
from eval_hpi_judge import SCORE_CONFIGS as HPI_SCORE_CONFIGS
from eval_hpi_judge import run_judge as run_hpi_judge
from judge_client import ensure_score_configs, invoke_judge
from models import HistoryOfPresentIllness, PEJudgeScore, PhysicalExam

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clinical_note_dataset_eval")

PE_JUDGE_PROMPT_PATH = Path("prompts/physical_exam_judge_prompt.txt")
DATASET_NAME = "clinical-note-eval-encounters"

# Golden data for data/encounter_{1,2}.txt, confirmed with a human reviewer.
# Only `physical_exam` has a golden reference: the HPI rubric grades against
# the transcript directly, so no reference HPI is needed for scoring.
GOLDEN_ITEMS = [
    {
        "id": "clinical-note-encounter-1",
        "transcript_path": Path("data/encounter_1.txt"),
        "expected_output": {
            "physical_exam": [
                {
                    "system": "general",
                    "findings": "Febrile / hot to touch on exam, despite a cold wound margin.",
                },
                {
                    "system": "skin",
                    "findings": (
                        "Wound edge pale, almost white, and cold to touch; pale streaking "
                        "extends from the wound toward the chest."
                    ),
                },
                {
                    "system": "neurologic",
                    "findings": "Patient unconscious throughout the exam; no response to stimulus on the left arm.",
                },
            ]
        },
    },
    {
        "id": "clinical-note-encounter-2",
        "transcript_path": Path("data/encounter_2.txt"),
        "expected_output": {
            "physical_exam": [
                {"system": "general", "findings": "Patient awake and alert."},
                {
                    "system": "skin",
                    "findings": "Superficial burns on both hands and on the face from ash/heat exposure.",
                },
                {
                    "system": "extremities",
                    "findings": (
                        "Hand wound (finger amputation) healing well; mild pain elicited on "
                        "manipulation during wound cleaning."
                    ),
                },
            ]
        },
    },
]

PE_SCORE_CONFIGS = [
    {
        "name": "pe_accuracy",
        "data_type": "NUMERIC",
        "min_value": 0,
        "max_value": 4,
        "description": "Faithfulness of non-placeholder PE findings to the transcript's exam. See prompts/physical_exam_judge_prompt.txt.",
    },
    {
        "name": "pe_completeness",
        "data_type": "NUMERIC",
        "min_value": 0,
        "max_value": 4,
        "description": "Depth of findings captured for each system the exam actually covered. See prompts/physical_exam_judge_prompt.txt.",
    },
    {
        "name": "pe_tone",
        "data_type": "NUMERIC",
        "min_value": 0,
        "max_value": 4,
        "description": "PE documentation register. See prompts/physical_exam_judge_prompt.txt.",
    },
    {
        "name": "pe_precision",
        "data_type": "NUMERIC",
        "min_value": 0,
        "max_value": 1,
        "description": "Of the body systems the extraction predicted as examined, the fraction actually in the golden exam set.",
    },
    {
        "name": "pe_recall",
        "data_type": "NUMERIC",
        "min_value": 0,
        "max_value": 1,
        "description": "Of the golden exam's body systems, the fraction the extraction predicted.",
    },
]

SCORE_CONFIGS = HPI_SCORE_CONFIGS + PE_SCORE_CONFIGS

# Boilerplate phrasing the extraction model emits for a system it padded in
# rather than actually examined (see `_is_filler`).
_FILLER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"no specific .*(findings|exam)",
        r"no .*(examination|exam|assessment) (was )?performed",
        r"no .*(findings|complaints) (noted|documented|stated)",
        r"not (examined|assessed|performed|documented)",
    ]
]


def _is_filler(findings_text: str) -> bool:
    """True if every sentence in `findings_text` is a boilerplate 'not examined' phrase.

    Splits on sentence boundaries rather than checking the whole string, since
    entries often mix one real observation with a trailing filler clause —
    that should count as real, not filler.

    Known limitation: this is a phrasing heuristic tuned against this
    project's extraction prompt/model, not a semantic classifier. A finding
    built entirely from patient self-report (e.g. "patient reports thirst; no
    abdominal tenderness noted on exam") can pass this check yet still be
    fabricated/misclassified — that case isn't caught by precision/recall at
    all, only by the LLM judge's Accuracy dimension.
    """
    text = (findings_text or "").strip()
    if not text:
        return True
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return all(any(p.search(s) for p in _FILLER_PATTERNS) for s in sentences)


def ensure_dataset(client) -> None:
    """Create the shared golden dataset and upsert its 2 items if not already present."""
    try:
        client.create_dataset(
            name=DATASET_NAME,
            description="Golden HPI + physical-exam data for data/encounter_1.txt and data/encounter_2.txt.",
        )
    except Exception as exc:
        logger.info("Dataset %s already exists (%s)", DATASET_NAME, exc)

    for golden in GOLDEN_ITEMS:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=golden["id"],
            input=golden["transcript_path"].read_text(encoding="utf-8"),
            expected_output=golden["expected_output"],
        )
    logger.info("Dataset %s ready with %d items", DATASET_NAME, len(GOLDEN_ITEMS))


# Populated by `main()` before the experiment runs; evaluators read it via
# closure so each Evaluation is pinned to its score config (otherwise scores
# land unranged in the UI — 0-4 for the judge scores, 0-1 for precision/recall).
_CONFIG_IDS: dict[str, str] = {}


async def run_clinical_note(*, item, **kwargs) -> dict:
    """Task: extract both the HPI narrative and PE findings for one item's transcript."""
    hpi_agent = build_agent(extraction_model, hpi_prompt, HistoryOfPresentIllness)
    pe_agent = build_agent(extraction_model, physical_exam_prompt, PhysicalExam)
    config = {"callbacks": get_callbacks()}
    hpi_result, pe_result = await asyncio.gather(
        run_agent(hpi_agent, item.input, config),
        run_agent(pe_agent, item.input, config),
    )
    return {
        "hpi": hpi_result.hpi,
        "physical_exam": [f.model_dump() for f in pe_result.findings],
    }


def hpi_judge_evaluator(*, input, output, **kwargs) -> list[Evaluation]:
    """LLM-judge evaluator: one call scoring HPI accuracy/completeness/tone against the transcript."""
    judge_prompt = HPI_JUDGE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    result, judge_model_name = run_hpi_judge(judge_prompt, input, output["hpi"])
    metadata = {"judge_model": judge_model_name}
    return [
        Evaluation(
            name="hpi_accuracy",
            value=result.accuracy_score,
            comment=result.accuracy_rationale,
            metadata=metadata,
            data_type="NUMERIC",
            config_id=_CONFIG_IDS.get("hpi_accuracy"),
        ),
        Evaluation(
            name="hpi_completeness",
            value=result.completeness_score,
            comment=result.completeness_rationale,
            metadata=metadata,
            data_type="NUMERIC",
            config_id=_CONFIG_IDS.get("hpi_completeness"),
        ),
        Evaluation(
            name="hpi_tone",
            value=result.tone_score,
            comment=result.tone_rationale,
            metadata=metadata,
            data_type="NUMERIC",
            config_id=_CONFIG_IDS.get("hpi_tone"),
        ),
    ]


def pe_precision_recall_evaluator(*, output, expected_output, **kwargs) -> list[Evaluation]:
    """Code evaluator: PE system-set precision/recall against the golden systems.

    A predicted system counts as a true positive only if its findings text is
    non-filler (see `_is_filler`) — the model padding in all 12 enum values
    with "not documented" placeholders should not inflate precision.
    """
    golden_systems = {item["system"] for item in (expected_output or {}).get("physical_exam", [])}
    predicted = (output or {}).get("physical_exam", [])
    predicted_systems = {entry["system"] for entry in predicted if not _is_filler(entry.get("findings"))}

    true_positives = predicted_systems & golden_systems
    precision = len(true_positives) / len(predicted_systems) if predicted_systems else 0.0
    recall = len(true_positives) / len(golden_systems) if golden_systems else 0.0
    comment = f"Predicted systems: {sorted(predicted_systems)}. Golden: {sorted(golden_systems)}."

    return [
        Evaluation(
            name="pe_precision",
            value=precision,
            comment=comment,
            data_type="NUMERIC",
            config_id=_CONFIG_IDS.get("pe_precision"),
        ),
        Evaluation(
            name="pe_recall",
            value=recall,
            comment=comment,
            data_type="NUMERIC",
            config_id=_CONFIG_IDS.get("pe_recall"),
        ),
    ]


def pe_judge_evaluator(*, input, output, **kwargs) -> list[Evaluation]:
    """LLM-judge evaluator: one call scoring PE accuracy/completeness/tone."""
    judge_prompt = PE_JUDGE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    pe_findings = (output or {}).get("physical_exam", [])
    pe_text = "\n".join(f"- {entry['system']}: {entry['findings']}" for entry in pe_findings)
    messages = [
        {"role": "system", "content": judge_prompt},
        {
            "role": "user",
            "content": f"<transcript>\n{input}\n</transcript>\n\n<physical_exam>\n{pe_text}\n</physical_exam>",
        },
    ]
    result, judge_model_name = invoke_judge(PEJudgeScore, messages)
    metadata = {"judge_model": judge_model_name}
    return [
        Evaluation(
            name="pe_accuracy",
            value=result.accuracy_score,
            comment=result.accuracy_rationale,
            metadata=metadata,
            data_type="NUMERIC",
            config_id=_CONFIG_IDS.get("pe_accuracy"),
        ),
        Evaluation(
            name="pe_completeness",
            value=result.completeness_score,
            comment=result.completeness_rationale,
            metadata=metadata,
            data_type="NUMERIC",
            config_id=_CONFIG_IDS.get("pe_completeness"),
        ),
        Evaluation(
            name="pe_tone",
            value=result.tone_score,
            comment=result.tone_rationale,
            metadata=metadata,
            data_type="NUMERIC",
            config_id=_CONFIG_IDS.get("pe_tone"),
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run the combined HPI + PE golden-dataset experiment.")
    parser.add_argument(
        "--run-name",
        help="Name for this experiment run (defaults to '<OLLAMA_MODEL>-<timestamp>' so runs are distinguishable by generator model in the Langfuse UI).",
    )
    args = parser.parse_args()

    client = get_client()
    global _CONFIG_IDS
    _CONFIG_IDS = ensure_score_configs(client, SCORE_CONFIGS)
    ensure_dataset(client)

    dataset = client.get_dataset(DATASET_NAME)
    result = dataset.run_experiment(
        name="Clinical note eval",
        run_name=args.run_name or f"{OLLAMA_MODEL}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
        description="LLM-judged HPI + PE accuracy/completeness/tone, plus PE system precision/recall.",
        task=run_clinical_note,
        evaluators=[hpi_judge_evaluator, pe_precision_recall_evaluator, pe_judge_evaluator],
        metadata={"generator_model": OLLAMA_MODEL},
    )
    print(result.format())

    client.flush()


if __name__ == "__main__":
    main()
