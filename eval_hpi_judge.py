"""Score a single generated HPI against its transcript with an LLM judge.

Reads the `hpi` observation from an existing Langfuse trace (its `input` is
the transcript; its `output` is `{"hpi": {"hpi": "..."}}`, LangChain's default
capture of the `hpi` node's return value — no extra instrumentation needed),
asks a judge model to score Accuracy, Completeness, and Tone (0-4 each, see
`prompts/hpi_judge_prompt.txt`), and attaches the three scores back to that
observation in Langfuse.

Judge model: Amazon Bedrock by default, falling back to a local Ollama model
(distinct from the `hpi` node's generator) if Bedrock is unreachable.
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client

from models import HPIJudgeScore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hpi_judge")

JUDGE_PROMPT_PATH = Path("prompts/hpi_judge_prompt.txt")
BEDROCK_MODEL_ID = os.getenv("JUDGE_BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
FALLBACK_OLLAMA_MODEL = os.getenv("JUDGE_FALLBACK_OLLAMA_MODEL", "mistral:latest")

SCORE_CONFIGS = [
    {
        "name": "hpi_accuracy",
        "description": "Faithfulness of the HPI to the transcript (0-4). See prompts/hpi_judge_prompt.txt.",
    },
    {
        "name": "hpi_completeness",
        "description": "Coverage of transcript content relevant to the present illness (0-4). See prompts/hpi_judge_prompt.txt.",
    },
    {
        "name": "hpi_tone",
        "description": "Physician-documentation register of the HPI (0-4). See prompts/hpi_judge_prompt.txt.",
    },
]


def latest_trace_id(client) -> str:
    """Return the most recent LangGraph trace's id."""
    traces = client.api.trace.list(name="LangGraph", limit=1, order_by="timestamp.desc")
    if not traces.data:
        raise ValueError("No LangGraph traces found in Langfuse")
    return traces.data[0].id


def find_hpi_observation(client, trace_id: str):
    """Return the `hpi` node's span from a trace."""
    trace = client.api.trace.get(trace_id)
    for obs in trace.observations:
        if obs.name == "hpi" and obs.type == "CHAIN":
            return obs
    raise ValueError(f"No 'hpi' observation found on trace {trace_id}")


def judge_hpi(model, judge_prompt: str, transcript: str, hpi: str) -> HPIJudgeScore:
    structured_model = model.with_structured_output(HPIJudgeScore)
    return structured_model.invoke(
        [
            {"role": "system", "content": judge_prompt},
            {
                "role": "user",
                "content": f"<transcript>\n{transcript}\n</transcript>\n\n<hpi>\n{hpi}\n</hpi>",
            },
        ]
    )


def run_judge(judge_prompt: str, transcript: str, hpi: str) -> tuple[HPIJudgeScore, str]:
    """Score with Bedrock; fall back to a local Ollama model if Bedrock fails.

    The fallback is a different model than the `hpi` node's generator
    (gemma4:e2b) so judge != generator even during an outage.
    """
    from langchain_aws import ChatBedrockConverse

    try:
        model = ChatBedrockConverse(model_id=BEDROCK_MODEL_ID, region_name=BEDROCK_REGION, temperature=0)
        result = judge_hpi(model, judge_prompt, transcript, hpi)
        return result, f"bedrock:{BEDROCK_MODEL_ID}"
    except Exception as exc:
        logger.warning(
            "Bedrock judge call failed (%s); falling back to local Ollama %s", exc, FALLBACK_OLLAMA_MODEL
        )
        from langchain_ollama import ChatOllama

        model = ChatOllama(model=FALLBACK_OLLAMA_MODEL, temperature=0)
        result = judge_hpi(model, judge_prompt, transcript, hpi)
        return result, f"ollama:{FALLBACK_OLLAMA_MODEL}"


def ensure_score_configs(client) -> dict[str, str]:
    """Create the 3 hpi_* score configs if they don't already exist."""
    existing = {c.name: c.id for c in client.api.score_configs.get().data}
    config_ids = {}
    for cfg in SCORE_CONFIGS:
        if cfg["name"] in existing:
            config_ids[cfg["name"]] = existing[cfg["name"]]
            continue
        created = client.api.score_configs.create(
            name=cfg["name"],
            data_type="NUMERIC",
            min_value=0,
            max_value=4,
            description=cfg["description"],
        )
        config_ids[cfg["name"]] = created.id
        logger.info("Created score config %s (%s)", cfg["name"], created.id)
    return config_ids


def main():
    parser = argparse.ArgumentParser(description="Score one HPI trace with an LLM judge.")
    parser.add_argument(
        "--trace-id",
        help="Langfuse trace id to score (defaults to the most recent LangGraph trace).",
    )
    args = parser.parse_args()

    client = get_client()
    trace_id = args.trace_id or latest_trace_id(client)
    logger.info("Scoring trace %s", trace_id)

    hpi_obs = find_hpi_observation(client, trace_id)
    transcript = hpi_obs.input["transcript"]
    # The node's return value `{"hpi": HistoryOfPresentIllness(hpi=...)}` is what
    # LangChain records as this span's output, hence the doubly-nested "hpi" key.
    hpi_text = (hpi_obs.output or {}).get("hpi", {}).get("hpi")
    if not hpi_text:
        raise ValueError(f"hpi observation {hpi_obs.id} has no usable output.hpi.hpi text")

    judge_prompt = JUDGE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    result, judge_model_name = run_judge(judge_prompt, transcript, hpi_text)
    logger.info("Judge model used: %s", judge_model_name)

    config_ids = ensure_score_configs(client)
    scores = [
        ("hpi_accuracy", result.accuracy_score, result.accuracy_rationale),
        ("hpi_completeness", result.completeness_score, result.completeness_rationale),
        ("hpi_tone", result.tone_score, result.tone_rationale),
    ]
    for name, value, rationale in scores:
        client.create_score(
            name=name,
            value=value,
            trace_id=trace_id,
            observation_id=hpi_obs.id,
            data_type="NUMERIC",
            config_id=config_ids[name],
            comment=rationale,
            metadata={"judge_model": judge_model_name},
        )
        logger.info("%s: %d/4 — %s", name, value, rationale)

    client.flush()
    logger.info("Scores attached to trace %s, observation %s", trace_id, hpi_obs.id)


if __name__ == "__main__":
    main()
