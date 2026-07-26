"""A minimal LangGraph medical-scribe agent.

Reads a patient-doctor transcript, uses Ollama to extract a structured
`ClinicalNote`, and saves it to disk via a tool.

Four focused sub-agents each extract one section of the note — history of present
illness (HPI), vitals, physical exam, and diagnoses — and their outputs are merged
into a single `ClinicalNote`. They are all built by one `build_agent(prompt,
response_format, tools)` helper and run sequentially as nodes of a LangGraph
`StateGraph`.

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
from langchain.agents.middleware import ModelRetryMiddleware, ModelCallLimitMiddleware
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
    CandidateExtraction,
    ClinicalNote,
    DiagnosesOutput,
    DiagnosisCandidate,
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


OUTPUT_PATH = Path("outputs/clinical_note.json")
SYSTEM_PROMPT_PATH = Path("prompts/system_prompt.txt")
DIAGNOSES_PROMPT_PATH = Path("prompts/diagnoses_prompt.txt")
DIAGNOSES_PROMPT_WITH_TOOL_PATH = Path("prompts/diagnoses_prompt_with_tool.txt")
DIAGNOSES_CANDIDATES_PROMPT_PATH = Path("prompts/diagnoses_candidates_prompt.txt")
DIAGNOSES_SELECTION_PROMPT_PATH = Path("prompts/diagnoses_selection_prompt.txt")
VITALS_PROMPT_PATH = Path("prompts/vitals_prompt.txt")
HPI_PROMPT_PATH = Path("prompts/hpi_prompt.txt")
PHYSICAL_EXAM_PROMPT_PATH = Path("prompts/physical_exam_prompt.txt")

# ICD-10-CM reference database used by search_icd10_codes/search_icd10_codes_batch.
# Defaults to the full code set; point this at a narrower parquet (see
# notebooks/focused_icd10_db.py) to compare search quality on a smaller,
# domain-scoped database.
ICD10_DB_PATH = os.getenv("ICD10_DB_PATH", "data/ICD10_DB.parquet")
logger.info("ICD-10 database: %s", ICD10_DB_PATH)


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
        A list of {"icd10_code", "description", "score"} candidates, already
        sorted best-first — rely on this order, not on the "score" value itself.
        "score" is an internal ranking signal, not a confidence percentage, and
        isn't on a fixed scale; close or low-looking scores don't mean the top
        result is a weak match.
    """
    return search_icd10_hybrid(diagnosis_name, limit=limit, path=ICD10_DB_PATH)


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
        {"icd10_code", "description", "score"} candidates, already sorted
        best-first — rely on this order, not on the "score" value itself.
        "score" is an internal ranking signal, not a confidence percentage, and
        isn't on a fixed scale; close or low-looking scores don't mean the top
        result is a weak match.
    """
    return search_icd10_hybrid_batch(diagnosis_names, limit=limit, path=ICD10_DB_PATH)


# The LLM: a local Ollama model. A low temperature keeps extraction near-deterministic.
# Both the model name and temperature are configurable via env vars (see .env.example).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
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
# Used by select_diagnoses_node (deterministic diagnoses workflow): reasoning
# on, temp=0/seeded. A/B tested against tool_model_no_reasoning on the same
# pre-fetched candidates/search_results: without reasoning, the model reliably
# (even at temp=0) kept only one of several well-matched candidates, dropping
# the rest outright rather than working through the full list — it isn't
# variance, it consistently can't hold all candidates in a single pass without
# a scratchpad. With reasoning on, it reliably finds the well-matched
# candidates instead; num_predict is higher to leave room for that reasoning.
select_model = ChatOllama(
    **shared,
    temperature=0.0,
    seed=42,
    reasoning=True,
    num_predict=8192,
)

# Number of times to retry a failed model call (e.g. an empty structured-output
# response) before giving up. Configurable via env var (see .env.example).
# ModelRetryMiddleware retries with exponential backoff on any exception.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
logger.info("Model call retries: %d", LLM_MAX_RETRIES)

# Hard ceiling on model calls per node, independent of LLM_MAX_RETRIES: when a
# tool-based structured-output agent replies without ever calling the hidden
# structuring tool (seen with the HPI agent, whose single free-text narrative
# field invites a prose answer instead of a tool call), create_agent's own loop
# just keeps re-invoking the model. That isn't an exception, so
# ModelRetryMiddleware never sees it and the loop can run for minutes. This
# caps it and fails fast instead.
MODEL_CALL_LIMIT = int(os.getenv("MODEL_CALL_LIMIT", "8"))

# A higher ceiling for the diagnoses agent when the ICD-10 tool is enabled: it
# genuinely needs more turns than the other nodes. Observed in Langfuse traces:
# the small model re-queries search_icd10_codes_batch repeatedly with
# progressively reworded diagnosis names (e.g. "necrotizing fasciitis of
# shoulder", then "necrotizing fasciitis", then "infectious necrotizing
# fasciitis of shoulder") instead of settling after one batched call, so 8+
# tool round trips before a final answer is normal, not a sign it's stuck.
DIAGNOSES_MODEL_CALL_LIMIT = int(os.getenv("DIAGNOSES_MODEL_CALL_LIMIT", "20"))


def get_middleware(call_limit: int = MODEL_CALL_LIMIT) -> list:
    """Shared middleware for every agent: retry failed model calls with backoff,
    and hard-cap total model calls so a stuck agent fails fast instead of
    looping for minutes.

    `call_limit` is overridable per agent: the tool-calling diagnoses agent
    needs a much higher ceiling than the single-shot extraction agents (see
    DIAGNOSES_MODEL_CALL_LIMIT).
    """
    return [
        ModelRetryMiddleware(max_retries=LLM_MAX_RETRIES),
        ModelCallLimitMiddleware(run_limit=call_limit, exit_behavior="error"),
    ]

logger.info("Loading system prompt from %s", SYSTEM_PROMPT_PATH)
system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
logger.info("Loading vitals prompt from %s", VITALS_PROMPT_PATH)
vitals_prompt = VITALS_PROMPT_PATH.read_text(encoding="utf-8").strip()
logger.info("Loading HPI prompt from %s", HPI_PROMPT_PATH)
hpi_prompt = HPI_PROMPT_PATH.read_text(encoding="utf-8").strip()
logger.info("Loading physical exam prompt from %s", PHYSICAL_EXAM_PROMPT_PATH)
physical_exam_prompt = PHYSICAL_EXAM_PROMPT_PATH.read_text(encoding="utf-8").strip()
logger.info("Loading diagnoses candidates prompt from %s", DIAGNOSES_CANDIDATES_PROMPT_PATH)
diagnoses_candidates_prompt = DIAGNOSES_CANDIDATES_PROMPT_PATH.read_text(encoding="utf-8").strip()
logger.info("Loading diagnoses selection prompt from %s", DIAGNOSES_SELECTION_PROMPT_PATH)
diagnoses_selection_prompt = DIAGNOSES_SELECTION_PROMPT_PATH.read_text(encoding="utf-8").strip()


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

    Inputs: transcript, diagnoses_prompt, use_tool, deterministic.
    Outputs (written by each node): hpi, vitals, physical_exam, diagnoses.
    """

    transcript: str
    diagnoses_prompt: str
    use_tool: bool
    deterministic: bool
    hpi: HistoryOfPresentIllness
    vitals: VitalSigns
    physical_exam: PhysicalExam
    diagnoses: DiagnosesOutput


def build_agent(
    llm,
    prompt: str,
    response_format,
    tools: list | None = None,
    call_limit: int = MODEL_CALL_LIMIT,
):
    """Build a scribe sub-agent from its task prompt and output schema.

    Every sub-agent shares the same model, scribe system role, and retry
    middleware; they differ only in their task `prompt`, `response_format`,
    optional `tools`, and (for diagnoses-with-tool) `call_limit`. This single
    builder keeps them consistent and makes adding a new agent a one-line call.

    Args:
        prompt: Task instructions for this agent, appended to the shared scribe
            system role to form the agent's system prompt.
        response_format: The structured-output target. Pass a bare Pydantic model
            to use the default tool-based strategy, or wrap it in
            `ProviderStrategy(schema=...)` to use Ollama's native JSON-schema
            `format` (grammar-constrained generation). The default strategy is fine
            for schemas that read as "extraction" (vitals, physical exam), but a
            schema whose field is really free-text prose (HPI's narrative) invites
            the model to just answer in plain content instead of calling the hidden
            structuring tool, which starves it of a tool call and can loop for
            minutes (see hpi_node) — use `ProviderStrategy` there instead. Native
            output is also preferred for the diagnoses agent: with tools, the
            default strategy makes structured output a hidden tool that competes
            with the domain tool and the small model tends to answer in prose
            (dropping the result); without tools, it lets the model emit slow
            free-form reasoning before the JSON.
        tools: Domain tools to expose, or None for a tool-free agent.
        call_limit: Passed to ModelCallLimitMiddleware (see get_middleware).
            Defaults to MODEL_CALL_LIMIT; the diagnoses node passes
            DIAGNOSES_MODEL_CALL_LIMIT instead when the ICD-10 tool is enabled,
            since that workflow legitimately needs more model calls.

    Returns:
        A compiled agent whose `invoke` returns the validated instance at
        `result["structured_response"]`.
    """
    return create_agent(
        model=llm,
        system_prompt=f"{system_prompt}\n\n{prompt}",
        tools=tools or [],
        response_format=response_format,
        middleware=get_middleware(call_limit=call_limit),
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
    return {"vitals": await run_agent(agent, state["transcript"], config)}


async def hpi_node(state: ScribeState, config: RunnableConfig) -> dict:
    """Write the History of Present Illness with a dedicated, tool-free agent.

    Uses `ProviderStrategy` (native JSON-schema output) rather than the default
    tool-based strategy: the HPI schema is a single free-text narrative field, and
    a bare Pydantic `response_format` asks the model to hand that narrative back
    via a hidden structuring tool call. The model tends to just answer in prose
    instead of calling it, so create_agent's loop keeps re-invoking the model
    hoping for a tool call — see ModelCallLimitMiddleware in get_middleware for
    the resulting failure mode. Native output sidesteps the tool call entirely.
    """
    logger.info("HPI node: writing history of present illness (tool-free)")
    agent = build_agent(extraction_model, hpi_prompt, ProviderStrategy(schema=HistoryOfPresentIllness))
    return {"hpi": await run_agent(agent, state["transcript"], config)}


async def physical_exam_node(state: ScribeState, config: RunnableConfig) -> dict:
    """Extract physical exam findings with a dedicated, tool-free agent."""
    logger.info("Physical exam node: extracting PE findings (tool-free)")
    agent = build_agent(extraction_model, physical_exam_prompt, PhysicalExam)
    return {"physical_exam": await run_agent(agent, state["transcript"], config)}


class DiagnosesWorkflowState(TypedDict):
    """State for the deterministic diagnoses workflow (see build_diagnoses_workflow)."""

    transcript: str
    candidates: list[DiagnosisCandidate]
    search_results: dict
    diagnoses: DiagnosesOutput


async def extract_candidates_node(state: DiagnosesWorkflowState, config: RunnableConfig) -> dict:
    """Extract candidate diagnoses from the transcript; no ICD-10 codes yet.

    Uses `extraction_model` (temperature=0, seeded) rather than
    `tool_model_no_reasoning`: at temperature=0.1 this step showed real
    run-to-run variance, occasionally diluting well-supported candidates (e.g.
    "gas gangrene", "necrotizing fasciitis") with generic trauma-differential
    boilerplate ("blunt force trauma", "ARDS", "hypovolemic shock from
    hemorrhage") the transcript didn't actually describe.
    """
    logger.info("Diagnoses workflow: extracting candidates (tool-free)")
    agent = build_agent(
        extraction_model,
        diagnoses_candidates_prompt,
        ProviderStrategy(schema=CandidateExtraction),
    )
    result = await run_agent(agent, state["transcript"], config)
    return {"candidates": result.candidates}


async def search_candidates_node(state: DiagnosesWorkflowState, config: RunnableConfig) -> dict:
    """Look up every candidate's search_term exactly once — deterministic, no LLM call."""
    terms = sorted({candidate.search_term for candidate in state["candidates"]})
    logger.info("Diagnoses workflow: searching ICD-10 for %d term(s)", len(terms))
    results = search_icd10_hybrid_batch(terms, limit=10, path=ICD10_DB_PATH)
    return {"search_results": results}


async def select_diagnoses_node(state: DiagnosesWorkflowState, config: RunnableConfig) -> dict:
    """Pick the best code per candidate from its pre-fetched results, then consolidate.

    Includes the original transcript alongside the candidates: without it, this
    step has no way to notice a candidate's search results are all poor matches
    (rather than reasonable choices) and just picks the least-bad one instead of
    omitting the candidate — seen in practice with a "penetrating thoracic
    injury" candidate whose only returned codes were things like a heart-lung
    transplant infection code, nowhere close to what the transcript describes.
    """
    logger.info("Diagnoses workflow: selecting codes from pre-fetched results")
    candidate_blocks = [
        f"- candidate_name: {candidate.candidate_name}\n"
        f"  section: {candidate.section}\n"
        f"  search_results: {json.dumps(state['search_results'].get(candidate.search_term, []))}"
        for candidate in state["candidates"]
    ]
    context = "\n".join(candidate_blocks) or "(no candidates extracted)"

    agent = build_agent(select_model, diagnoses_selection_prompt, ProviderStrategy(schema=DiagnosesOutput))
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"<transcript>\n{state['transcript']}\n</transcript>\n\n"
                        f"<candidates>\n{context}\n</candidates>"
                    ),
                }
            ]
        },
        config=config,
        stream=False,
    )
    diagnoses: DiagnosesOutput = result["structured_response"]

    # Enforce "each code appears in exactly one list" in code rather than solely
    # via prompt instruction: the model still duplicated a code across both
    # lists occasionally even after that rule was added to the prompt. Keep the
    # assessment entry (every encounter must have one) and drop the duplicate
    # from differentials.
    assessment_codes = {d.icd10_code for d in diagnoses.assessment}
    diagnoses.differential_diagnoses = [
        d for d in diagnoses.differential_diagnoses if d.icd10_code not in assessment_codes
    ]
    return {"diagnoses": diagnoses}


def build_diagnoses_workflow():
    """The deterministic diagnoses pipeline: extract candidates, search once, then select.

    Unlike diagnoses_node's tool-calling agent, the model never decides whether
    or how many times to search: search_candidates_node calls the ICD-10 lookup
    directly, exactly once, independent of either LLM step's behavior. This
    trades the tool-calling agent's flexibility (it can search again if a first
    pass looks weak) for a hard guarantee against the retry/repeat-call failure
    modes that motivated it.
    """
    builder = StateGraph(DiagnosesWorkflowState)
    builder.add_node("extract_candidates", extract_candidates_node)
    builder.add_node("search_candidates", search_candidates_node)
    builder.add_node("select_diagnoses", select_diagnoses_node)
    builder.add_edge(START, "extract_candidates")
    builder.add_edge("extract_candidates", "search_candidates")
    builder.add_edge("search_candidates", "select_diagnoses")
    builder.add_edge("select_diagnoses", END)
    return builder.compile()


_diagnoses_workflow = build_diagnoses_workflow()


async def diagnoses_node(state: ScribeState, config: RunnableConfig) -> dict:
    """Extract and (optionally) tool-validate the differentials and assessment.

    Uses `ProviderStrategy` (native JSON-schema output) rather than the default
    tool-based strategy so the structured result survives alongside the ICD-10 tool.

    With the tool enabled, uses DIAGNOSES_MODEL_CALL_LIMIT instead of the
    default MODEL_CALL_LIMIT: the small model iteratively re-queries the ICD-10
    tool with reworded diagnosis names rather than converging in one or two
    calls, so it legitimately needs more turns than the other nodes.

    When both the tool and `deterministic` are set, delegates to the
    deterministic workflow above instead (see build_diagnoses_workflow).
    """
    use_tool = state["use_tool"]
    if use_tool and state.get("deterministic"):
        logger.info("Diagnoses node: deterministic workflow (extract -> search once -> select)")
        result = await _diagnoses_workflow.ainvoke({"transcript": state["transcript"]}, config=config)
        return {"diagnoses": result["diagnoses"]}

    logger.info(
        "Diagnoses node: extracting diagnoses, ICD-10 tool %s",
        "enabled" if use_tool else "disabled",
    )
    tools = [search_icd10_codes_batch] if use_tool else []
    agent = build_agent(
        tool_model if use_tool else tool_model_no_reasoning,
        state["diagnoses_prompt"],
        ProviderStrategy(schema=DiagnosesOutput),
        call_limit=DIAGNOSES_MODEL_CALL_LIMIT if use_tool else MODEL_CALL_LIMIT,
        tools=tools,
    )
    return {"diagnoses": await run_agent(agent, state["transcript"], config)}


ALL_NODES = ["hpi", "vitals", "physical_exam", "diagnoses"]

# Each node reads only `transcript`/`diagnoses_prompt`/`use_tool`/`deterministic`
# from the initial state, never another node's output, so any subset of
# ALL_NODES can run safely.
_NODE_SPECS = {
    "hpi": (hpi_node, hpi_prompt),
    "vitals": (vitals_node, vitals_prompt),
    "physical_exam": (physical_exam_node, physical_exam_prompt),
    "diagnoses": (diagnoses_node, None),
}


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
    deterministic: bool = False,
    cache: bool = False,
    nodes: list[str] | None = None,
) -> ClinicalNote:
    """Run the scribe graph and merge its outputs into a ClinicalNote.

    The HPI, vitals, and physical exam nodes (tool-free) and the diagnoses node
    (ICD-10 tool when `use_tool`) run sequentially (one Ollama call at a time);
    their results are combined here.

    `deterministic` only matters when `use_tool` is set: it swaps the diagnoses
    node's tool-calling agent for the deterministic extract/search/select
    workflow (see build_diagnoses_workflow).

    `nodes` restricts which of ALL_NODES actually run (default: all four) — handy
    for fast iteration on a single node without paying for the others. Any node
    that didn't run contributes its schema default (or an empty list) to
    the returned ClinicalNote instead of a real result.

    With `cache=True`, node results are cached to disk (see build_cache), so a
    later run with unchanged inputs skips the LLM calls for the unchanged nodes.
    """
    nodes = nodes if nodes is not None else ALL_NODES
    cache_backend = build_cache() if cache else None
    if cache_backend is not None:
        logger.info("Node caching enabled (DiskCache at %s, ttl=%s)", CACHE_DIR, CACHE_TTL)
    scribe_graph = build_sequential_graph(cache_backend, nodes)
    logger.info(
        "Running scribe graph over transcript (%d chars), nodes=%s, ICD-10 tool %s%s",
        len(transcript),
        ",".join(nodes),
        "enabled" if use_tool else "disabled",
        " (deterministic)" if use_tool and deterministic else "",
    )
    start = time.perf_counter()
    final = await scribe_graph.ainvoke(
        {
            "transcript": transcript,
            "diagnoses_prompt": diagnoses_prompt,
            "use_tool": use_tool,
            "deterministic": deterministic,
        },
        config={"callbacks": get_callbacks()},
        stream=False,
    )
    logger.info("Graph finished in %.2fs", time.perf_counter() - start)

    hpi_result = final.get("hpi")
    vitals_result = final.get("vitals")
    pe_result = final.get("physical_exam")
    diagnoses_result = final.get("diagnoses")
    return ClinicalNote(
        hpi=hpi_result.hpi if hpi_result else "",
        vitals=vitals_result or VitalSigns(),
        physical_exam=pe_result.findings if pe_result else [],
        differential_diagnoses=diagnoses_result.differential_diagnoses if diagnoses_result else [],
        assessment=diagnoses_result.assessment if diagnoses_result else [],
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
        help=f"Path to the transcript text file.",
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
        "--deterministic",
        action="store_true",
        help=(
            "Only meaningful with --use-tool. Replaces the diagnoses node's "
            "tool-calling agent with a fixed extract/search/select pipeline: the "
            "model extracts candidates, the ICD-10 lookup runs directly in code "
            "exactly once, then the model selects codes from those results — no "
            "model-driven decision about whether or how many times to search."
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
        deterministic=args.deterministic,
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
