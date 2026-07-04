"""A minimal LangGraph medical-scribe agent.

Reads a patient-doctor transcript, uses Ollama (qwen3.5:4b) to extract a
structured `ClinicalNote`, and saves it to disk via a tool.

Structured output is handled natively by `create_agent` through the
`response_format=` parameter: the validated Pydantic instance is returned at
`result["structured_response"]`.

Prereqs:
  - Ollama running locally with the model pulled: `ollama pull qwen3.5:4b`
  - Deps: langchain, langgraph, langchain-ollama (see pyproject.toml)
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from icd10_search import search_icd10
from models import ClinicalNote

# Load variables from a local .env file if present (see .env.example).
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scribe")


def get_callbacks() -> list:
    """Return LangChain callbacks for tracing.

    Enables Langfuse only when its keys are set, so the script stays runnable
    with no tracing configured. Set the LANGFUSE_* vars (see .env.example) to
    turn it on.
    """
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        logger.info("Langfuse keys not set; running without tracing")
        return []

    from langfuse.langchain import CallbackHandler

    logger.info("Langfuse tracing enabled")
    return [CallbackHandler()]

TRANSCRIPT_PATH = Path("data/transcript.txt")
OUTPUT_PATH = Path("outputs/clinical_note.json")
SYSTEM_PROMPT_PATH = Path("prompts/system_prompt.txt")
USER_PROMPT_PATH = Path("prompts/user_prompt.txt")


@tool
def save_json(filename: str, data: dict) -> str:
    """Save a dictionary to a JSON file on disk.

    Args:
        filename: Name (or path) of the file to write, e.g. "output.json".
        data: The JSON-serializable object to store.
    """
    path = Path(filename)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return f"Saved JSON to {path.resolve()}"


@tool
def search_icd10_codes(diagnosis_name: str, limit: int = 10) -> list[dict]:
    """Fuzzy-search active ICD-10-CM codes by diagnosis name.

    Use this to find the correct ICD-10-CM code for a condition mentioned in the
    transcript. Returns the closest matching codes, best match first.

    Args:
        diagnosis_name: The condition to look up, e.g. "acute bronchitis".
        limit: Maximum number of candidate codes to return.

    Returns:
        A list of {"icd10_code", "description", "score"} candidates.
    """
    return search_icd10(diagnosis_name, limit=limit)


# The LLM: a local Ollama model. temperature=0 keeps extraction deterministic.
logger.info("Creating Ollama model 'qwen3.5:4b'")
llm = ChatOllama(model="qwen3.5:4b", temperature=0)

# The scribe agent. `response_format=ClinicalNote` makes the agent return a
# validated ClinicalNote instance in result["structured_response"].
# No tools here on purpose: giving a small model both a tool and a
# response_format tends to make it call the tool instead of emitting the
# structured output. We persist the result separately (see __main__).
logger.info("Loading system prompt from %s", SYSTEM_PROMPT_PATH)
system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

logger.info("Creating scribe agent")
scribe = create_agent(
    model=llm,
    system_prompt=system_prompt,
    response_format=ClinicalNote,
)
logger.info("Scribe agent ready")


def extract_note(transcript: str, user_prompt: str) -> ClinicalNote:
    """Run the scribe agent over a transcript and return the structured note.

    The user prompt (extraction instructions) is combined with the transcript
    into the user message.
    """
    logger.info("Running agent over transcript (%d chars)", len(transcript))
    start = time.perf_counter()
    result = scribe.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{user_prompt}\n\n<transcript>\n{transcript}\n</transcript>",
                }
            ]
        },
        config={"callbacks": get_callbacks()},
    )
    logger.info("Agent finished in %.2fs", time.perf_counter() - start)
    return result["structured_response"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a structured clinical note from a patient-doctor transcript."
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        type=Path,
        default=TRANSCRIPT_PATH,
        help=f"Path to the transcript text file (default: {TRANSCRIPT_PATH}).",
    )
    parser.add_argument(
        "-u",
        "--user-prompt",
        type=Path,
        default=USER_PROMPT_PATH,
        help=f"Path to the user prompt (extraction instructions) file (default: {USER_PROMPT_PATH}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Path to write the JSON note (default: {OUTPUT_PATH}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    total_start = time.perf_counter()

    if not args.transcript.exists():
        raise SystemExit(f"Transcript not found: {args.transcript}")
    if not args.user_prompt.exists():
        raise SystemExit(f"User prompt not found: {args.user_prompt}")

    logger.info("Reading transcript from %s", args.transcript)
    transcript = args.transcript.read_text(encoding="utf-8")

    logger.info("Reading user prompt from %s", args.user_prompt)
    user_prompt = args.user_prompt.read_text(encoding="utf-8").strip()

    note = extract_note(transcript, user_prompt)
    print(note.model_dump_json(indent=2))

    # Persist the structured note (structured_response) to the output path.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json.invoke({"filename": str(args.output), "data": note.model_dump()})
    logger.info("Saved response to %s", args.output)

    # Flush any pending Langfuse traces before the script exits.
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse import get_client

        get_client().flush()
        logger.info("Flushed Langfuse traces")

    logger.info("Total time: %.2fs", time.perf_counter() - total_start)
