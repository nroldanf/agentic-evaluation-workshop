"""A minimal LangGraph medical-scribe agent.

Reads a patient-doctor transcript, uses Ollama to extract a
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
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from icd10_search import search_icd10, search_icd10_batch
from models import ClinicalNote, DiagnosesOutput, VitalSigns

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
DIAGNOSES_PROMPT_PATH = Path("prompts/diagnoses_prompt.txt")
DIAGNOSES_PROMPT_WITH_TOOL_PATH = Path("prompts/diagnoses_prompt_with_tool.txt")
VITALS_PROMPT_PATH = Path("prompts/vitals_prompt.txt")


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


@tool
def search_icd10_codes_batch(diagnosis_names: list[str], limit: int = 10) -> dict:
    """Fuzzy-search active ICD-10-CM codes for many diagnosis names at once.

    Call this ONCE with every diagnosis name you need to code. Returns, for each
    input name, the closest matching codes (best match first).

    Args:
        diagnosis_names: Condition names to look up, e.g. ["acute bronchitis", "asthma"].
        limit: Maximum number of candidate codes to return per name.

    Returns:
        A mapping of each diagnosis name to its list of
        {"icd10_code", "description", "score"} candidates.
    """
    return search_icd10_batch(diagnosis_names, limit=limit)


# The LLM: a local Ollama model. temperature=0 keeps extraction deterministic.
# The model name is configurable via the OLLAMA_MODEL env var (see .env.example).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
logger.info("Creating Ollama model %r", OLLAMA_MODEL)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

logger.info("Loading system prompt from %s", SYSTEM_PROMPT_PATH)
system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
logger.info("Loading vitals prompt from %s", VITALS_PROMPT_PATH)
vitals_prompt = VITALS_PROMPT_PATH.read_text(encoding="utf-8").strip()


class ScribeState(TypedDict):
    """State for the scribe graph.

    Inputs: transcript, diagnoses_prompt, use_tool.
    Outputs (written by parallel nodes): vital_signs, diagnoses.
    """

    transcript: str
    diagnoses_prompt: str
    use_tool: bool
    vital_signs: VitalSigns
    diagnoses: DiagnosesOutput


def build_diagnoses_agent(use_tool: bool):
    """Build the diagnoses agent, giving it the ICD-10 tool only when requested.

    Output schema is `DiagnosesOutput` (differentials + assessment) — vitals are
    handled by a separate node. The structured-output strategy is chosen per mode
    because the 4B model is reliable with a different one in each:
      - No tools: the default tool-based strategy (pass the schema directly).
      - With tools: `ProviderStrategy` (Ollama's native JSON-schema `format`). The
        default strategy makes structured output a hidden tool that competes with
        the domain tool, and the small model tends to answer in prose instead —
        dropping the structured result. Native output coerces the final answer
        regardless of tool usage.
    """
    tools = [search_icd10_codes_batch] if use_tool else []
    response_format = (
        ProviderStrategy(schema=DiagnosesOutput) if use_tool else DiagnosesOutput
    )
    return create_agent(
        model=llm,
        system_prompt=system_prompt,
        tools=tools,
        response_format=response_format,
    )


def build_vitals_agent():
    """Build the vitals agent (no tools).

    Output schema is `VitalSigns`. Uses the default tool-based structured-output
    strategy, which is reliable for a tool-free agent on this model.
    """
    return create_agent(
        model=llm,
        system_prompt=system_prompt,
        response_format=VitalSigns,
    )


def vitals_node(state: ScribeState, config: RunnableConfig) -> dict:
    """Extract vitals with a dedicated, tool-free agent.

    Kept separate from diagnoses because the 4B model reliably drops vitals when
    it is also doing the ICD-10 tool workflow in the same pass.
    """
    logger.info("Vitals node: extracting vitals (tool-free)")
    agent = build_vitals_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{vitals_prompt}\n\n<transcript>\n{state['transcript']}\n</transcript>",
                }
            ]
        },
        config=config,
    )
    return {"vital_signs": result["structured_response"]}


def diagnoses_node(state: ScribeState, config: RunnableConfig) -> dict:
    """Extract and (optionally) tool-validate the differentials and assessment."""
    use_tool = state["use_tool"]
    logger.info(
        "Diagnoses node: extracting diagnoses, ICD-10 tool %s",
        "enabled" if use_tool else "disabled",
    )
    agent = build_diagnoses_agent(use_tool)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{state['diagnoses_prompt']}\n\n<transcript>\n{state['transcript']}\n</transcript>",
                }
            ]
        },
        config=config,
    )
    return {"diagnoses": result["structured_response"]}


def build_parallel_graph():
    """Vitals and diagnoses run in parallel, then merge.

    Needs a local Ollama server configured for concurrency (OLLAMA_NUM_PARALLEL >= 2);
    otherwise the two simultaneous structured-output calls can return empty responses.
    Kept for when the server is configured for parallelism.
    """
    builder = StateGraph(ScribeState)
    builder.add_node("vitals", vitals_node)
    builder.add_node("diagnoses", diagnoses_node)
    # Fan-out from START (parallel), fan-in to END. The nodes write different
    # state keys, so no reducers are needed.
    builder.add_edge(START, "vitals")
    builder.add_edge(START, "diagnoses")
    builder.add_edge("vitals", END)
    builder.add_edge("diagnoses", END)
    return builder.compile()


def build_sequential_graph():
    """Vitals then diagnoses, one Ollama call at a time, then merge.

    Reliable against a single local Ollama server (no concurrent requests). The
    nodes write different state keys, so no reducers are needed.
    """
    builder = StateGraph(ScribeState)
    builder.add_node("vitals", vitals_node)
    builder.add_node("diagnoses", diagnoses_node)
    builder.add_edge(START, "vitals")
    builder.add_edge("vitals", "diagnoses")
    builder.add_edge("diagnoses", END)
    return builder.compile()


def extract_note(
    transcript: str,
    diagnoses_prompt: str,
    use_tool: bool = False,
    parallel: bool = False,
) -> ClinicalNote:
    """Run the scribe graph and merge its parallel outputs into a ClinicalNote.

    The vitals node (tool-free) and the diagnoses node (ICD-10 tool when
    `use_tool`) run either sequentially (default, single local Ollama server) or
    in parallel (`parallel=True`, requires OLLAMA_NUM_PARALLEL >= 2); their
    results are combined here.
    """
    scribe_graph = (
        build_parallel_graph() if parallel else build_sequential_graph()
    )
    logger.info(
        "Running scribe graph (%s) over transcript (%d chars), ICD-10 tool %s",
        "parallel" if parallel else "sequential",
        len(transcript),
        "enabled" if use_tool else "disabled",
    )
    start = time.perf_counter()
    final = scribe_graph.invoke(
        {
            "transcript": transcript,
            "diagnoses_prompt": diagnoses_prompt,
            "use_tool": use_tool,
        },
        config={"callbacks": get_callbacks()},
    )
    logger.info("Graph finished in %.2fs", time.perf_counter() - start)

    diagnoses = final["diagnoses"]
    return ClinicalNote(
        vital_signs=final["vital_signs"],
        differential_diagnoses=diagnoses.differential_diagnoses,
        assessment=diagnoses.assessment,
    )


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
        "-d",
        "--diagnoses-prompt",
        type=Path,
        default=None,
        help=(
            "Path to the diagnoses prompt (extraction instructions) file. Defaults to "
            f"{DIAGNOSES_PROMPT_PATH}, or {DIAGNOSES_PROMPT_WITH_TOOL_PATH} when --use-tool is set."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Path to write the JSON note (default: {OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--use-tool",
        action="store_true",
        help=(
            "Give the agent the ICD-10 search tool to validate diagnosis codes. "
            "Defaults the diagnoses prompt to "
            f"{DIAGNOSES_PROMPT_WITH_TOOL_PATH}."
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Run the vitals and diagnoses nodes in parallel instead of "
            "sequentially. Requires a local Ollama server configured for "
            "concurrency (OLLAMA_NUM_PARALLEL >= 2)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    total_start = time.perf_counter()

    # Resolve the diagnoses prompt: honor -d if given, else pick the default that
    # matches the flow (tool-based vs baseline).
    diagnoses_prompt_path = args.diagnoses_prompt or (
        DIAGNOSES_PROMPT_WITH_TOOL_PATH if args.use_tool else DIAGNOSES_PROMPT_PATH
    )

    if not args.transcript.exists():
        raise SystemExit(f"Transcript not found: {args.transcript}")
    if not diagnoses_prompt_path.exists():
        raise SystemExit(f"Diagnoses prompt not found: {diagnoses_prompt_path}")

    logger.info("Reading transcript from %s", args.transcript)
    transcript = args.transcript.read_text(encoding="utf-8")

    logger.info("Reading diagnoses prompt from %s", diagnoses_prompt_path)
    diagnoses_prompt = diagnoses_prompt_path.read_text(encoding="utf-8").strip()

    note = extract_note(
        transcript, diagnoses_prompt, use_tool=args.use_tool, parallel=args.parallel
    )
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
