"""A minimal LangGraph medical-scribe agent.

Reads a patient-doctor transcript, uses Ollama to extract a structured
`ClinicalNote`, and saves it to disk via a tool.

Four focused sub-agents each extract one section of the note — history of present
illness (HPI), vitals, physical exam, and diagnoses — and their outputs are merged
into a single `ClinicalNote`. They are all built by one `build_agent(prompt,
response_format, tools)` helper and run as nodes of a LangGraph `StateGraph`,
either sequentially (default) or in parallel (`--parallel`).

Structured output is handled natively by `create_agent` through the
`response_format=` parameter: the validated Pydantic instance is returned at
`result["structured_response"]`.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy

import models as models_module
from cache import DiskCache
from icd10_search import search_icd10_hybrid, search_icd10_hybrid_batch
from models import (
    Assessment,
    ClinicalNote,
    DiagnosesOutput,
    HistoryOfPresentIllness,
    PhysicalExam,
    VitalSigns,
)

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
HPI_PROMPT_PATH = Path("prompts/hpi_prompt.txt")
PHYSICAL_EXAM_PROMPT_PATH = Path("prompts/physical_exam_prompt.txt")


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
def search_icd10_codes(diagnosis_name: str, limit: int = 25) -> list[dict]:
    """Search active ICD-10-CM codes by diagnosis name.

    Use this to find the correct ICD-10-CM code for a condition mentioned in the
    transcript. Returns the closest matching codes, best match first.

    Args:
        diagnosis_name: The condition to look up, e.g. "acute bronchitis".
        limit: Maximum number of candidate codes to return.

    Returns:
        A list of {"icd10_code", "description", "score"} candidates.
    """
    return search_icd10_hybrid(diagnosis_name, limit=limit)


@tool
def search_icd10_codes_batch(diagnosis_names: list[str], limit: int = 25) -> dict:
    """Search active ICD-10-CM codes for many diagnosis names at once.

    Search for all the diagnosis names in `diagnosis_names` and return a mapping of each name to its list of candidate codes, best match first. Useful for validating a list of
    differentials extracted from a transcript.

    Args:
        diagnosis_names: Condition names to look up, e.g. ["acute bronchitis", "asthma"].
        limit: Maximum number of candidate codes to return per name.

    Returns:
        A mapping of each diagnosis name to its list of
        {"icd10_code", "description", "score"} candidates.
    """
    return search_icd10_hybrid_batch(diagnosis_names, limit=limit)


# The LLM: a local Ollama model. A low temperature keeps extraction near-deterministic.
# Both the model name and temperature are configurable via env vars (see .env.example).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
# Per-request timeout in seconds for calls to the Ollama server. Small local
# models can be slow, so this defaults high; raise it further via env var if a
# model still times out (see .env.example).
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
logger.info(
    "Creating Ollama model %r (temperature=%s, timeout=%ss)",
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
)

shared = {
    "model": OLLAMA_MODEL,
    "num_ctx": 20480,
    "keep_alive": "15m",
    "validate_model_on_init": True,
    "client_kwargs": {
        "timeout": httpx.Timeout(
            connect=10.0,
            read=OLLAMA_TIMEOUT,
            write=30.0,
            pool=10.0,
        )
    },
}

extraction_model = ChatOllama(
    **shared,
    temperature=0.0,
    seed=42,
    reasoning=False,
    num_predict=1024,
)
tool_model = ChatOllama(
    **shared,
    temperature=0.1,
    reasoning=True,
    num_predict=4096,
)
# Same params as tool_model, minus reasoning: used when use_tool is False so
# reasoning is never enabled without the ICD-10 tool.
tool_model_no_reasoning = ChatOllama(
    **shared,
    temperature=0.1,
    reasoning=False,
    num_predict=4096,
)

# Number of times to retry a failed model call (e.g. an empty structured-output
# response) before giving up. Configurable via env var (see .env.example).
# ModelRetryMiddleware retries with exponential backoff on any exception.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
logger.info("Model call retries: %d", LLM_MAX_RETRIES)


def get_middleware() -> list:
    """Shared middleware for every agent: retry failed model calls with backoff."""
    return [ModelRetryMiddleware(max_retries=LLM_MAX_RETRIES)]

logger.info("Loading system prompt from %s", SYSTEM_PROMPT_PATH)
system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
logger.info("Loading vitals prompt from %s", VITALS_PROMPT_PATH)
vitals_prompt = VITALS_PROMPT_PATH.read_text(encoding="utf-8").strip()
logger.info("Loading HPI prompt from %s", HPI_PROMPT_PATH)
hpi_prompt = HPI_PROMPT_PATH.read_text(encoding="utf-8").strip()
logger.info("Loading physical exam prompt from %s", PHYSICAL_EXAM_PROMPT_PATH)
physical_exam_prompt = PHYSICAL_EXAM_PROMPT_PATH.read_text(encoding="utf-8").strip()


# --- Node-level caching (opt-in via --cache) ------------------------------
# A DiskCache persists each node's result across processes, so re-running with
# unchanged inputs skips the (slow, local) LLM call. The cache key includes the
# transcript, the node's effective prompt, the model, and the temperature (see
# make_cache_key), so editing a prompt or switching models re-runs only the
# affected node(s); the rest stay warm. Configurable via env vars (see .env.example).
CACHE_DIR = os.getenv("LANGGRAPH_CACHE_DIR", ".cache/scribe")
_cache_ttl_env = os.getenv("LANGGRAPH_CACHE_TTL", "").strip()
# Empty/unset means no expiry: a hit is served only for truly identical inputs.
CACHE_TTL = int(_cache_ttl_env) if _cache_ttl_env else None

# Types the cache's serde may (de)serialize: every class defined in `models`
# (the Pydantic result models AND enums they nest, e.g. PhysicalExamSystem).
_MODEL_ALLOWLIST = [
    ("models", name)
    for name in dir(models_module)
    if isinstance(getattr(models_module, name), type)
    and getattr(models_module, name).__module__ == "models"
]


def build_cache() -> DiskCache:
    """Build the persistent disk-cache backend for node-level caching."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=_MODEL_ALLOWLIST)
    return DiskCache(CACHE_DIR, serde=serde)


def make_cache_key(task_prompt: str | None = None):
    """Build a node cache-key function keyed on the inputs that affect its output.

    Keys on the transcript, the node's effective prompt (shared system role + task
    prompt), the model, and the temperature — plus the ICD-10 tool flag for the
    diagnoses node. LangGraph namespaces cache entries per node, so editing one
    prompt busts only that node's cache; the others stay warm.

    Pass the fixed task prompt for the HPI/vitals/physical-exam nodes; pass None
    for the diagnoses node, whose prompt and tool flag come from the run state.
    """

    def key_func(val: dict) -> str:
        if task_prompt is None:  # diagnoses node: prompt + tool flag are in state
            prompt = val.get("diagnoses_prompt", "")
            use_tool = val.get("use_tool", False)
        else:
            prompt, use_tool = task_prompt, False
        payload = {
            "transcript": val.get("transcript", ""),
            "system_prompt": system_prompt,
            "prompt": prompt,
            "use_tool": use_tool,
            "model": OLLAMA_MODEL,
            "temperature": OLLAMA_TEMPERATURE,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    return key_func


def _cache_policy(enabled: bool, task_prompt: str | None) -> CachePolicy | None:
    """A CachePolicy for a node when caching is enabled, else None (no caching)."""
    if not enabled:
        return None
    return CachePolicy(ttl=CACHE_TTL, key_func=make_cache_key(task_prompt))


class ScribeState(TypedDict):
    """State for the scribe graph.

    Inputs: transcript, diagnoses_prompt, use_tool.
    Outputs (written by parallel nodes): hpi, vital_signs, physical_exam, diagnoses.
    """

    transcript: str
    diagnoses_prompt: str
    use_tool: bool
    hpi: HistoryOfPresentIllness
    vital_signs: VitalSigns
    physical_exam: PhysicalExam
    diagnoses: DiagnosesOutput


def build_agent(llm, prompt: str, response_format, tools: list | None = None):
    """Build a scribe sub-agent from its task prompt and output schema.

    Every sub-agent shares the same model, scribe system role, and retry
    middleware; they differ only in their task `prompt`, `response_format`, and
    optional `tools`. This single builder keeps them consistent and makes adding a
    new agent a one-line call.

    Args:
        prompt: Task instructions for this agent, appended to the shared scribe
            system role to form the agent's system prompt.
        response_format: The structured-output target. Pass a bare Pydantic model
            to use the default tool-based strategy (reliable for tool-free agents),
            or wrap it in `ProviderStrategy(schema=...)` to use Ollama's native
            JSON-schema `format` (grammar-constrained generation). Native output is
            preferred for the diagnoses agent: with tools, the default strategy makes
            structured output a hidden tool that competes with the domain tool and
            the small model tends to answer in prose (dropping the result); without
            tools, it lets the model emit slow free-form reasoning before the JSON.
        tools: Domain tools to expose, or None for a tool-free agent.

    Returns:
        A compiled agent whose `invoke` returns the validated instance at
        `result["structured_response"]`.
    """
    return create_agent(
        model=llm,
        system_prompt=f"{system_prompt}\n\n{prompt}",
        tools=tools or [],
        response_format=response_format,
        middleware=get_middleware(),
    )


async def run_agent(agent, transcript: str, config: RunnableConfig):
    """Invoke a scribe sub-agent over the transcript and return its structured output."""
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"<transcript>\n{transcript}\n</transcript>",
                }
            ]
        },
        config=config,
        stream=False,
    )
    return result["structured_response"]


async def vitals_node(state: ScribeState, config: RunnableConfig) -> dict:
    """Extract vitals with a dedicated, tool-free agent.

    Kept separate from diagnoses because the 4B model reliably drops vitals when
    it is also doing the ICD-10 tool workflow in the same pass.
    """
    logger.info("Vitals node: extracting vitals (tool-free)")
    agent = build_agent(extraction_model, vitals_prompt, VitalSigns)
    return {"vital_signs": await run_agent(agent, state["transcript"], config)}


async def hpi_node(state: ScribeState, config: RunnableConfig) -> dict:
    """Write the History of Present Illness with a dedicated, tool-free agent."""
    logger.info("HPI node: writing history of present illness (tool-free)")
    agent = build_agent(extraction_model, hpi_prompt, HistoryOfPresentIllness)
    return {"hpi": await run_agent(agent, state["transcript"], config)}


async def physical_exam_node(state: ScribeState, config: RunnableConfig) -> dict:
    """Extract physical exam findings with a dedicated, tool-free agent."""
    logger.info("Physical exam node: extracting PE findings (tool-free)")
    agent = build_agent(extraction_model, physical_exam_prompt, PhysicalExam)
    return {"physical_exam": await run_agent(agent, state["transcript"], config)}


async def diagnoses_node(state: ScribeState, config: RunnableConfig) -> dict:
    """Extract and (optionally) tool-validate the differentials and assessment.

    Uses `ProviderStrategy` (native JSON-schema output) rather than the default
    tool-based strategy so the structured result survives alongside the ICD-10 tool.
    """
    use_tool = state["use_tool"]
    logger.info(
        "Diagnoses node: extracting diagnoses, ICD-10 tool %s",
        "enabled" if use_tool else "disabled",
    )
    tools = [search_icd10_codes_batch] if use_tool else []
    agent = build_agent(
        tool_model if use_tool else tool_model_no_reasoning,
        state["diagnoses_prompt"],
        ProviderStrategy(schema=DiagnosesOutput),
        tools=tools,
    )
    return {"diagnoses": await run_agent(agent, state["transcript"], config)}


ALL_NODES = ["hpi", "vitals", "physical_exam", "diagnoses"]

# Each node reads only `transcript`/`diagnoses_prompt`/`use_tool` from the initial
# state, never another node's output, so any subset of ALL_NODES can run safely.
_NODE_SPECS = {
    "hpi": (hpi_node, hpi_prompt),
    "vitals": (vitals_node, vitals_prompt),
    "physical_exam": (physical_exam_node, physical_exam_prompt),
    "diagnoses": (diagnoses_node, None),
}


def build_parallel_graph(cache=None, nodes: list[str] | None = None):
    """The selected nodes run in parallel, then merge (default: all four).

    Needs a local Ollama server configured for concurrency (OLLAMA_NUM_PARALLEL >= 4);
    otherwise the simultaneous structured-output calls can return empty responses.
    Kept for when the server is configured for parallelism.

    When a `cache` backend is passed, each node caches its result under a policy
    keyed on that node's inputs (see make_cache_key).
    """
    nodes = nodes if nodes is not None else ALL_NODES
    enabled = cache is not None
    builder = StateGraph(ScribeState)
    # Fan-out from START (parallel), fan-in to END. The nodes write different
    # state keys, so no reducers are needed.
    for name in nodes:
        fn, prompt = _NODE_SPECS[name]
        builder.add_node(name, fn, cache_policy=_cache_policy(enabled, prompt))
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    return builder.compile(cache=cache)


def build_sequential_graph(cache=None, nodes: list[str] | None = None):
    """The selected nodes run in order, one Ollama call at a time (default: all four).

    Reliable against a single local Ollama server (no concurrent requests). The
    nodes write different state keys, so no reducers are needed.

    When a `cache` backend is passed, each node caches its result under a policy
    keyed on that node's inputs (see make_cache_key).
    """
    nodes = nodes if nodes is not None else ALL_NODES
    enabled = cache is not None
    builder = StateGraph(ScribeState)
    for name in nodes:
        fn, prompt = _NODE_SPECS[name]
        builder.add_node(name, fn, cache_policy=_cache_policy(enabled, prompt))

    # Connect the selected nodes sequentially, in canonical order.
    prev = START
    for name in nodes:
        builder.add_edge(prev, name)
        prev = name
    builder.add_edge(prev, END)
    return builder.compile(cache=cache)


async def extract_note(
    transcript: str,
    diagnoses_prompt: str,
    use_tool: bool = False,
    parallel: bool = False,
    cache: bool = False,
    nodes: list[str] | None = None,
) -> ClinicalNote:
    """Run the scribe graph and merge its parallel outputs into a ClinicalNote.

    The HPI, vitals, and physical exam nodes (tool-free) and the diagnoses node
    (ICD-10 tool when `use_tool`) run either sequentially (default, single local
    Ollama server) or in parallel (`parallel=True`, requires OLLAMA_NUM_PARALLEL
    >= 4); their results are combined here.

    `nodes` restricts which of ALL_NODES actually run (default: all four) — handy
    for fast iteration on a single node without paying for the others. Any node
    that didn't run contributes its schema default (or an empty Assessment) to
    the returned ClinicalNote instead of a real result.

    With `cache=True`, node results are cached to disk (see build_cache), so a
    later run with unchanged inputs skips the LLM calls for the unchanged nodes.
    """
    nodes = nodes if nodes is not None else ALL_NODES
    cache_backend = build_cache() if cache else None
    if cache_backend is not None:
        logger.info("Node caching enabled (DiskCache at %s, ttl=%s)", CACHE_DIR, CACHE_TTL)
    scribe_graph = (
        build_parallel_graph(cache_backend, nodes)
        if parallel
        else build_sequential_graph(cache_backend, nodes)
    )
    logger.info(
        "Running scribe graph (%s) over transcript (%d chars), nodes=%s, ICD-10 tool %s",
        "parallel" if parallel else "sequential",
        len(transcript),
        ",".join(nodes),
        "enabled" if use_tool else "disabled",
    )
    start = time.perf_counter()
    final = await scribe_graph.ainvoke(
        {
            "transcript": transcript,
            "diagnoses_prompt": diagnoses_prompt,
            "use_tool": use_tool,
        },
        config={
            "callbacks": get_callbacks(),
            "max_concurrency": 1,
        },
        stream=False,
    )
    logger.info("Graph finished in %.2fs", time.perf_counter() - start)

    hpi_result = final.get("hpi")
    vitals_result = final.get("vital_signs")
    pe_result = final.get("physical_exam")
    diagnoses_result = final.get("diagnoses")
    return ClinicalNote(
        hpi=hpi_result.hpi if hpi_result else "",
        vital_signs=vitals_result or VitalSigns(),
        physical_exam=pe_result.findings if pe_result else [],
        differential_diagnoses=diagnoses_result.differential_diagnoses if diagnoses_result else [],
        assessment=diagnoses_result.assessment
        if diagnoses_result
        else Assessment(icd10_code="", diagnosis_name=""),
    )


def _node_list(raw: str) -> list[str]:
    """Parse a comma-separated `--only`/`--skip` value into validated node names."""
    names = [n.strip() for n in raw.split(",") if n.strip()]
    invalid = [n for n in names if n not in ALL_NODES]
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown node(s) {invalid}; choose from {ALL_NODES}")
    return names


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
            "Run the HPI, vitals, physical exam, and diagnoses nodes in parallel "
            "instead of sequentially. Requires a local Ollama server configured for "
            "concurrency (OLLAMA_NUM_PARALLEL >= 4)."
        ),
    )
    node_group = parser.add_mutually_exclusive_group()
    node_group.add_argument(
        "--only",
        type=_node_list,
        metavar="NODE[,NODE...]",
        help=(
            "Run only these nodes, skipping the rest (comma-separated, from "
            f"{', '.join(ALL_NODES)}). Useful for fast iteration on one node. "
            "Nodes not run contribute an empty/default section to the output."
        ),
    )
    node_group.add_argument(
        "--skip",
        type=_node_list,
        metavar="NODE[,NODE...]",
        help=(
            "Run all nodes except these (comma-separated, from "
            f"{', '.join(ALL_NODES)})."
        ),
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help=(
            "Cache each node's result to disk so a later run with unchanged inputs "
            "(same transcript, prompt, and model) skips its LLM call. Editing a "
            f"prompt re-runs only that node. Cache dir: {CACHE_DIR}."
        ),
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help=f"Delete all cached node results (in {CACHE_DIR}) and exit.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()

    if args.clear_cache:
        build_cache().clear()
        logger.info("Cleared node cache at %s", CACHE_DIR)
        return

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

    if args.only:
        nodes = args.only
    elif args.skip:
        nodes = [n for n in ALL_NODES if n not in args.skip]
    else:
        nodes = ALL_NODES

    note = await extract_note(
        transcript,
        diagnoses_prompt,
        use_tool=args.use_tool,
        parallel=args.parallel,
        cache=args.cache,
        nodes=nodes,
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


if __name__ == "__main__":
    asyncio.run(main())
