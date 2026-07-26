import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building a Medical-Scribe Agent

    Notebook 1 of the workshop series. We reconstruct the scribe agent
    from `agent.py` and `models.py` piece by piece, running each piece
    **live** against one real transcript — no pre-computed outputs.
    *Evaluating* what it produces is notebook 2's job; here we only care
    about how the agent is built.

    We deliberately don't `import agent`: it reads its prompts relative
    to the repo root (which breaks once this notebook's cwd is
    `notebooks/`) and it pings the Ollama server the moment it's
    imported, to validate the model. Instead we rebuild the same pieces
    here, one at a time — that reconstruction *is* the lesson.
    """)
    return


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    repo_root = Path("..").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    load_dotenv(repo_root / ".env")
    return os, repo_root


@app.cell
def _():
    from typing import TypedDict

    import httpx
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
    from langchain.agents.structured_output import ProviderStrategy
    from langchain.tools import tool
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, StateGraph

    return (
        ChatOllama,
        END,
        ModelCallLimitMiddleware,
        ModelRetryMiddleware,
        ProviderStrategy,
        START,
        StateGraph,
        TypedDict,
        create_agent,
        httpx,
        tool,
    )


@app.cell
def _():
    import models
    from icd10_search import search_icd10_hybrid_batch

    return models, search_icd10_hybrid_batch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 — The schema is the contract

    Everything the agent produces is shaped by the Pydantic models in
    `models.py`, not by prompt wording alone. `create_agent(response_format=...)`
    uses a model's fields — names, types, and descriptions — to constrain what
    the LLM is allowed to return. So before touching a prompt, look at the
    schema it has to fill in.
    """)
    return


@app.cell
def _(mo, models):
    mo.accordion(
        {
            "VitalSigns": mo.json(models.VitalSigns.model_json_schema()),
            "HistoryOfPresentIllness": mo.json(models.HistoryOfPresentIllness.model_json_schema()),
            "PhysicalExam": mo.json(models.PhysicalExam.model_json_schema()),
            "DiagnosesOutput (differentials + assessment)": mo.json(
                models.DiagnosesOutput.model_json_schema()
            ),
            "ClinicalNote (the final merged output)": mo.json(models.ClinicalNote.model_json_schema()),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A design choice worth pausing on: `assessment` is `list[Assessment]`, not a
    single `Assessment` — the same shape as `differential_diagnoses`. A clinician
    can be actively working up more than one condition at once, so the primary
    diagnoses get the same "sorted by likelihood, top 3" treatment as the
    differentials, just extracted for a different purpose (working diagnosis vs.
    possibility considered). `Diagnosis` is the shared base model for both.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 — The transcript

    Everything below runs against one real patient-doctor transcript:
    encounter `RIV-001`, Frodo Baggins examined by Elrond.
    """)
    return


@app.cell
def _(repo_root):
    transcript_path = repo_root / "data" / "encounter_riv001.txt"
    transcript = transcript_path.read_text(encoding="utf-8")
    return (transcript,)


@app.cell
def _(mo, transcript):
    mo.plain_text(transcript)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3 — One sub-agent, end to end (HPI)

    `create_agent(model=..., system_prompt=..., tools=..., response_format=...)`
    is the whole recipe: a chat model, a system prompt (a shared scribe role +
    one task's instructions), optional tools, and the Pydantic schema to
    constrain the output. `result["structured_response"]` holds the validated
    instance.

    We start with the History of Present Illness (HPI) node: tool-free, one
    prompt, one schema — the simplest of the four.

    One wrinkle: HPI's schema is a single free-text narrative field (see the
    schema above). With the default (tool-based) structured-output strategy,
    the model is supposed to hand that narrative back via a hidden structuring
    tool call — but a plain narrative reads like a normal chat answer, so the
    model tends to just write it as prose instead of calling the tool. Nothing
    then tells the graph the turn is "done," so it keeps re-invoking the model.
    So, like the diagnoses node in Part 5, HPI uses `ProviderStrategy(schema=...)`
    — Ollama's native JSON-schema `format` — instead, sidestepping the tool
    call entirely.
    """)
    return


@app.cell
def _(ChatOllama, httpx, os):
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))

    # A low temperature keeps extraction near-deterministic; used by every
    # tool-free node (HPI, vitals, physical exam).
    extraction_model = ChatOllama(
        model=OLLAMA_MODEL,
        num_ctx=20480,
        keep_alive="15m",
        validate_model_on_init=True,
        temperature=0.0,
        seed=42,
        reasoning=False,
        num_predict=1024,
        client_kwargs={
            "timeout": httpx.Timeout(connect=10.0, read=OLLAMA_TIMEOUT, write=30.0, pool=10.0)
        },
    )
    return OLLAMA_MODEL, OLLAMA_TIMEOUT, extraction_model


@app.cell
def _(OLLAMA_MODEL, mo):
    mo.callout(
        mo.md(
            f"Every cell below makes a **live call** to the local Ollama model "
            f"`{OLLAMA_MODEL}` — there is no cached or pre-computed fallback. "
            f"Make sure `ollama serve` is running and the model is pulled "
            f"(`ollama pull {OLLAMA_MODEL}`) before running this notebook."
        ),
        kind="warn",
    )
    return


@app.cell
def _(repo_root):
    system_prompt = (repo_root / "prompts" / "system_prompt.txt").read_text(encoding="utf-8").strip()
    return (system_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Tracing (optional).** Same as `agent.py`'s `get_callbacks()`: Langfuse
    tracing turns on only when `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are
    set (see `.env.example`), so the notebook runs fine with no tracing
    configured. When they are set, every agent call below is traced and
    visible in the Langfuse UI.
    """)
    return


@app.cell
def _(os):
    def get_callbacks() -> list:
        """Return LangChain callbacks for tracing (Langfuse if configured, else none)."""
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            return []

        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]

    return (get_callbacks,)


@app.cell
def _(
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    create_agent,
    get_callbacks,
):
    def build_agent(llm, system_prompt, task_prompt, response_format, tools=None, call_limit=8):
        """Build a scribe sub-agent from its task prompt and output schema.

        Every sub-agent shares the same model-call recipe, differing only in
        its task `prompt`, `response_format`, optional `tools`, and (for
        diagnoses-with-tool) `call_limit` — this is the same `build_agent`
        helper `agent.py` uses for all four nodes.

        `ModelCallLimitMiddleware` is a fail-fast safety net: a tool-based
        structured-output agent that never calls its hidden structuring tool
        (see the HPI node below) loops inside create_agent without raising an
        exception, so `ModelRetryMiddleware` alone never catches it. Capping
        model calls turns that silent multi-minute loop into a clear error.

        The default of 8 fits the single-shot extraction agents. The
        tool-calling diagnoses agent (Part 6) needs a much higher `call_limit`:
        the small model iteratively re-queries the ICD-10 tool with reworded
        diagnosis names rather than converging in one or two calls, so 8 is
        reached before it ever produces a final answer — see Part 6.
        """
        return create_agent(
            model=llm,
            system_prompt=f"{system_prompt}\n\n{task_prompt}",
            tools=tools or [],
            response_format=response_format,
            middleware=[
                ModelRetryMiddleware(max_retries=5),
                ModelCallLimitMiddleware(run_limit=call_limit, exit_behavior="error"),
            ],
        )

    async def run_agent(agent, transcript):
        """Invoke a scribe sub-agent over the transcript, return its structured output."""
        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"}
                ]
            },
            config={"callbacks": get_callbacks()},
            stream=False,
        )
        return result["structured_response"]

    return build_agent, run_agent


@app.cell
def _(repo_root):
    hpi_prompt = (repo_root / "prompts" / "hpi_prompt.txt").read_text(encoding="utf-8").strip()
    return (hpi_prompt,)


@app.cell
async def _(
    ProviderStrategy,
    build_agent,
    extraction_model,
    hpi_prompt,
    models,
    run_agent,
    system_prompt,
    transcript,
):
    hpi_agent = build_agent(
        extraction_model,
        system_prompt,
        hpi_prompt,
        ProviderStrategy(schema=models.HistoryOfPresentIllness),
    )
    hpi_result = await run_agent(hpi_agent, transcript)
    hpi_result
    return hpi_agent, hpi_result


@app.cell
def _(hpi_result, mo):
    mo.md(f"**Generated HPI:**\n\n{hpi_result.hpi}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4 — Why four focused sub-agents, not one

    It would be simpler to ask one agent for the whole `ClinicalNote` in a
    single pass. In practice, a small local model asked to fill a big schema
    while also running a tool tends to drop fields — e.g. it reliably drops
    vitals when it's also busy validating ICD-10 codes in the same pass. So
    `agent.py` splits the note into four focused nodes — HPI, vitals, physical
    exam, diagnoses — each with a narrow prompt and a narrow schema, and merges
    their outputs afterward. We just built the HPI node above; vitals and
    physical exam follow the exact same pattern.
    """)
    return


@app.cell
async def _(
    build_agent,
    extraction_model,
    models,
    repo_root,
    run_agent,
    system_prompt,
    transcript,
):
    vitals_prompt = (repo_root / "prompts" / "vitals_prompt.txt").read_text(encoding="utf-8").strip()
    vitals_agent = build_agent(extraction_model, system_prompt, vitals_prompt, models.VitalSigns)
    vitals_result = await run_agent(vitals_agent, transcript)
    vitals_result
    return (vitals_agent,)


@app.cell
async def _(
    build_agent,
    extraction_model,
    models,
    repo_root,
    run_agent,
    system_prompt,
    transcript,
):
    physical_exam_prompt = (
        (repo_root / "prompts" / "physical_exam_prompt.txt").read_text(encoding="utf-8").strip()
    )
    physical_exam_agent = build_agent(
        extraction_model, system_prompt, physical_exam_prompt, models.PhysicalExam
    )
    physical_exam_result = await run_agent(physical_exam_agent, transcript)
    physical_exam_result
    return (physical_exam_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 5 — Diagnoses without a tool (the baseline)

    Before adding the ICD-10 tool in Part 6, look at the diagnoses agent
    without one — this is `agent.py`'s default (`--use-tool` is opt-in). The
    model assigns ICD-10-CM codes from its own training knowledge, using
    `prompts/diagnoses_prompt.txt` — the tool-less variant of the diagnoses
    instructions, as opposed to `diagnoses_prompt_with_tool.txt`.

    Even without a tool, this node still uses `ProviderStrategy(schema=...)`
    rather than the default tool-based strategy, the same native JSON-schema
    reasoning as HPI in Part 3. The difference here is the model: `agent.py`
    swaps in a *non-reasoning* model (`reasoning=False`) for this path. With
    reasoning enabled and no tool to call, the model tends to emit a long
    free-form reasoning preamble before finally producing the JSON — slow, and
    unnecessary when there's no tool result to reason about.
    """)
    return


@app.cell
def _(ChatOllama, OLLAMA_MODEL, OLLAMA_TIMEOUT, httpx):
    # Same params as tool_model (Part 6), minus reasoning: used without the
    # ICD-10 tool so reasoning is never enabled when there's no tool result to
    # reason about.
    tool_model_no_reasoning = ChatOllama(
        model=OLLAMA_MODEL,
        num_ctx=20480,
        keep_alive="15m",
        validate_model_on_init=True,
        temperature=0.1,
        reasoning=False,
        num_predict=4096,
        client_kwargs={
            "timeout": httpx.Timeout(connect=10.0, read=OLLAMA_TIMEOUT, write=30.0, pool=10.0)
        },
    )
    return (tool_model_no_reasoning,)


@app.cell
def _(repo_root):
    diagnoses_prompt_no_tool = (
        (repo_root / "prompts" / "diagnoses_prompt.txt").read_text(encoding="utf-8").strip()
    )
    return (diagnoses_prompt_no_tool,)


@app.cell
async def _(
    ProviderStrategy,
    build_agent,
    diagnoses_prompt_no_tool,
    models,
    run_agent,
    system_prompt,
    tool_model_no_reasoning,
    transcript,
):
    diagnoses_no_tool_agent = build_agent(
        tool_model_no_reasoning,
        system_prompt,
        diagnoses_prompt_no_tool,
        ProviderStrategy(schema=models.DiagnosesOutput),
    )
    diagnoses_no_tool_result = await run_agent(diagnoses_no_tool_agent, transcript)
    diagnoses_no_tool_result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 6 — Diagnoses, and the ICD-10 tool

    Now add the ICD-10 tool to the same diagnoses task: the model must not
    invent ICD-10-CM codes from memory, so it's given a
    `search_icd10_codes_batch` tool and instructed to assign only codes the
    tool actually returned (see `prompts/diagnoses_prompt_with_tool.txt`).

    This node also uses `ProviderStrategy(schema=...)` instead of the default
    (tool-based) structured-output strategy. With the default strategy,
    structured output is itself implemented as a hidden tool call — which then
    competes with the ICD-10 tool, and a small model tends to answer in prose
    instead, silently dropping the structured result. `ProviderStrategy` uses
    Ollama's native JSON-schema `format` (grammar-constrained generation)
    instead, so the domain tool and the structured output no longer compete for
    the same "call a tool" decision. Unlike the no-tool baseline in Part 5,
    reasoning is turned back on (`tool_model`, below) — the agent now has an
    actual tool result to reason about before answering.

    One more difference from Part 5: this agent is built with a higher
    `call_limit` (20, vs. `build_agent`'s default of 8). The small model tends
    to re-query the ICD-10 tool several times with progressively reworded
    diagnosis names rather than settling after one batched call, so 8 model
    calls can be exhausted before it ever produces a final answer — not
    because it's stuck, but because this workflow genuinely takes more turns.
    """)
    return


@app.cell
def _(repo_root, search_icd10_hybrid_batch, tool):
    icd10_path = str(repo_root / "data" / "ICD10_DB.parquet")

    @tool
    def search_icd10_codes_batch(diagnosis_names: list[str], limit: int = 25) -> dict:
        """Search active ICD-10-CM codes for many diagnosis names at once.

        Returns a mapping of each name to its list of {"icd10_code",
        "description", "score"} candidates, best match first.
        """
        return search_icd10_hybrid_batch(diagnosis_names, limit=limit, path=icd10_path)

    return (search_icd10_codes_batch,)


@app.cell
def _(ChatOllama, OLLAMA_MODEL, OLLAMA_TIMEOUT, httpx):
    # Same params as extraction_model, but with reasoning on and more headroom
    # for output tokens — the diagnoses node reasons through a multi-step
    # tool workflow (see diagnoses_prompt_with_tool.txt).
    tool_model = ChatOllama(
        model=OLLAMA_MODEL,
        num_ctx=20480,
        keep_alive="15m",
        validate_model_on_init=True,
        temperature=0.1,
        reasoning=True,
        num_predict=4096,
        client_kwargs={
            "timeout": httpx.Timeout(connect=10.0, read=OLLAMA_TIMEOUT, write=30.0, pool=10.0)
        },
    )
    return (tool_model,)


@app.cell
def _(repo_root):
    diagnoses_prompt = (
        (repo_root / "prompts" / "diagnoses_prompt_with_tool.txt").read_text(encoding="utf-8").strip()
    )
    return (diagnoses_prompt,)


@app.cell
def _(
    ProviderStrategy,
    build_agent,
    diagnoses_prompt,
    models,
    search_icd10_codes_batch,
    system_prompt,
    tool_model,
):
    diagnoses_agent = build_agent(
        tool_model,
        system_prompt,
        diagnoses_prompt,
        ProviderStrategy(schema=models.DiagnosesOutput),
        tools=[search_icd10_codes_batch],
        call_limit=20,
    )
    return (diagnoses_agent,)


@app.cell
async def _(diagnoses_agent, get_callbacks, transcript):
    # Call .ainvoke directly (not the run_agent helper) so we can inspect the
    # full message trace below, including the ICD-10 tool call the agent makes.
    diagnoses_full_result = await diagnoses_agent.ainvoke(
        {"messages": [{"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"}]},
        config={"callbacks": get_callbacks()},
        stream=False,
    )
    diagnoses_result = diagnoses_full_result["structured_response"]
    diagnoses_result
    return (diagnoses_full_result,)


@app.cell
def _(mo):
    mo.md("""
    **ICD-10 tool calls made during the run above:**
    """)
    return


@app.cell
def _(diagnoses_full_result):
    [
        {"tool": message.name, "result": message.content}
        for message in diagnoses_full_result["messages"]
        if getattr(message, "type", None) == "tool"
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 7 — Composing the nodes into a graph

    Four independent sub-agents aren't yet an agent pipeline. `agent.py` wires
    them into a LangGraph `StateGraph`: a typed state (`ScribeState`) that each
    node reads `transcript` from and writes one field to, run sequentially
    (one Ollama call at a time — reliable against a single local server).

    We reuse the exact sub-agents built above (`hpi_agent`, `vitals_agent`,
    `physical_exam_agent`, and the ICD-10 tool-enabled `diagnoses_agent` from
    Part 6, not the baseline from Part 5) as the graph's nodes — composing them
    is just wiring, no new extraction logic.
    """)
    return


@app.cell
def _(TypedDict, models):
    class ScribeState(TypedDict):
        """State threaded through the scribe graph: one input, four outputs."""

        transcript: str
        hpi: models.HistoryOfPresentIllness
        vitals: models.VitalSigns
        physical_exam: models.PhysicalExam
        diagnoses: models.DiagnosesOutput

    return (ScribeState,)


@app.cell
def _(
    diagnoses_agent,
    hpi_agent,
    physical_exam_agent,
    run_agent,
    vitals_agent,
):
    async def hpi_node(state):
        return {"hpi": await run_agent(hpi_agent, state["transcript"])}

    async def vitals_node(state):
        return {"vitals": await run_agent(vitals_agent, state["transcript"])}

    async def physical_exam_node(state):
        return {"physical_exam": await run_agent(physical_exam_agent, state["transcript"])}

    async def diagnoses_node(state):
        return {"diagnoses": await run_agent(diagnoses_agent, state["transcript"])}

    return diagnoses_node, hpi_node, physical_exam_node, vitals_node


@app.cell
def _(
    END,
    START,
    ScribeState,
    StateGraph,
    diagnoses_node,
    hpi_node,
    physical_exam_node,
    vitals_node,
):
    _builder = StateGraph(ScribeState)
    for _name, _fn in [
        ("hpi", hpi_node),
        ("vitals", vitals_node),
        ("physical_exam", physical_exam_node),
        ("diagnoses", diagnoses_node),
    ]:
        _builder.add_node(_name, _fn)

    _prev = START
    for _name in ("hpi", "vitals", "physical_exam", "diagnoses"):
        _builder.add_edge(_prev, _name)
        _prev = _name
    _builder.add_edge(_prev, END)

    scribe_graph = _builder.compile()
    return (scribe_graph,)


@app.cell
async def _(get_callbacks, scribe_graph, transcript):
    final_state = await scribe_graph.ainvoke(
        {"transcript": transcript},
        config={"callbacks": get_callbacks()},
    )
    return (final_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 8 — The assembled `ClinicalNote`

    `extract_note()` in `agent.py` does exactly this merge: pull each node's
    result out of the final graph state and assemble one `ClinicalNote`.
    """)
    return


@app.cell
def _(final_state, models):
    clinical_note = models.ClinicalNote(
        hpi=final_state["hpi"].hpi,
        vitals=final_state["vitals"],
        physical_exam=final_state["physical_exam"].findings,
        differential_diagnoses=final_state["diagnoses"].differential_diagnoses,
        assessment=final_state["diagnoses"].assessment,
    )
    clinical_note
    return (clinical_note,)


@app.cell
def _(clinical_note, mo):
    mo.json(clinical_note.model_dump())
    return


@app.cell
def _(os):
    # Flush any pending Langfuse traces before moving on, same as agent.py's
    # main() — otherwise buffered spans from this session may not be sent.
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse import get_client

        get_client().flush()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next: notebook 2 — evaluating what the agent produces

    This notebook only builds; it never asks whether the output is *good*.
    That's `evals/`: deterministic checks (`eval_diagnoses.py`, `eval_vitals.py`,
    `eval_trajectory.py`) comparing extracted output against
    `data/golden/golden_encounter_*.json`, plus LLM-as-a-judge scoring
    (`evals/eval_hpi_judge.py`, `evals/eval_clinical_note_dataset.py`) for the
    free-text sections a deterministic diff can't grade. See the README's
    "Evals" section for how to run each one — or run the CLI directly on this
    same transcript:

    ```
    uv run python agent.py data/encounter_riv001.txt --use-tool -o outputs/encounter_riv001.json
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
