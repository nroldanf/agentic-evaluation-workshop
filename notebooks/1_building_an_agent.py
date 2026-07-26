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
    # Construyendo un Agente Scribe Médico

    Notebook 1 de la serie del taller. Reconstruimos el agente scribe a partir
    de `agent.py` y `models.py` pieza por pieza, ejecutando cada pieza **en
    vivo** contra una transcripción real — sin resultados precalculados.
    *Evaluar* lo que produce es tarea del notebook 2; acá solo nos importa
    cómo está construido el agente.

    Deliberadamente no hacemos `import agent`: ese módulo lee sus prompts de
    forma relativa a la raíz del repo (lo cual se rompe en cuanto el cwd de
    este notebook es `notebooks/`) y hace ping al servidor de Ollama apenas se
    importa, para validar el modelo. En su lugar, reconstruimos acá las mismas
    piezas, una por una — esa reconstrucción *es* la lección.
    """)
    return


@app.cell
def _():
    import json
    import os
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    repo_root = Path("..").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    load_dotenv(repo_root / ".env")
    return json, os, repo_root


@app.cell
def _():
    from typing import TypedDict

    import httpx
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
    from langchain.agents.structured_output import ProviderStrategy
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
    )


@app.cell
def _():
    import models
    from icd10_search import search_icd10_hybrid_batch

    return models, search_icd10_hybrid_batch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parte 1 — El esquema es el contrato

    Todo lo que produce el agente está moldeado por los modelos de Pydantic en
    `models.py`, no solo por la redacción del prompt. `create_agent(response_format=...)`
    usa los campos de un modelo — nombres, tipos y descripciones — para
    restringir lo que el LLM puede devolver. Así que antes de tocar un prompt,
    hay que mirar el esquema que tiene que completar.
    """)
    return


@app.cell
def _(mo, models):
    mo.accordion(
        {
            "VitalSigns": mo.json(models.VitalSigns.model_json_schema()),
            "HistoryOfPresentIllness": mo.json(models.HistoryOfPresentIllness.model_json_schema()),
            "PhysicalExam": mo.json(models.PhysicalExam.model_json_schema()),
            "DiagnosesOutput (diferenciales + valoración)": mo.json(
                models.DiagnosesOutput.model_json_schema()
            ),
            "ClinicalNote (la salida final combinada)": mo.json(models.ClinicalNote.model_json_schema()),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Una decisión de diseño que vale la pena mirar con detenimiento:
    `assessment` es `list[Assessment]`, no un único `Assessment` — la misma
    forma que `differential_diagnoses`. Un clínico puede estar evaluando
    activamente más de una condición a la vez, así que los diagnósticos
    primarios reciben el mismo tratamiento de "ordenados por probabilidad, top
    3" que los diferenciales, solo que extraídos con un propósito distinto
    (diagnóstico de trabajo vs. posibilidad considerada). `Diagnosis` es el
    modelo base compartido por ambos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parte 2 — La transcripción

    Todo lo que sigue corre contra una transcripción real de un encuentro
    paciente-doctor: el encuentro `RIV-001`, Frodo Baggins examinado por
    Elrond.
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
    ## Parte 3 — Un subagente, de punta a punta (HPI)

    `create_agent(model=..., system_prompt=..., tools=..., response_format=...)`
    es toda la receta: un chat model, un system prompt (un rol scribe
    compartido + las instrucciones de una tarea), herramientas opcionales, y
    el esquema de Pydantic que restringe la salida. `result["structured_response"]`
    contiene la instancia ya validada.

    Empezamos con el nodo de Historia de la Enfermedad Actual (HPI): sin
    herramientas, un prompt, un esquema — el más simple de los cuatro.

    Un detalle: el esquema del HPI es un único campo narrativo de texto libre
    (ver el esquema de arriba). Con la estrategia de salida estructurada por
    defecto (basada en herramientas), se supone que el modelo devuelve esa
    narrativa mediante una llamada oculta a una herramienta de
    estructuración — pero una narrativa simple se lee como una respuesta de
    chat normal, así que el modelo tiende a escribirla directamente como
    prosa en vez de llamar a la herramienta. Como nada le avisa al grafo que
    el turno terminó, sigue invocando al modelo una y otra vez. Por eso, igual
    que el nodo de diagnósticos en la Parte 5, el HPI usa
    `ProviderStrategy(schema=...)` — el `format` nativo de JSON-schema de
    Ollama — en lugar de eso, evitando por completo la llamada a herramienta.
    """)
    return


@app.cell
def _(ChatOllama, httpx, os):
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))

    # Una temperature baja mantiene la extracción casi determinista; se usa en
    # todos los nodos sin herramientas (HPI, signos vitales, examen físico).
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
            f"Cada celda de acá abajo hace una **llamada en vivo** al modelo "
            f"Ollama local `{OLLAMA_MODEL}` — no hay ningún fallback cacheado "
            f"ni precalculado. Asegurate de tener `ollama serve` corriendo y "
            f"el modelo descargado (`ollama pull {OLLAMA_MODEL}`) antes de "
            f"ejecutar este notebook."
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
    **Tracing (opcional).** Igual que `get_callbacks()` en `agent.py`: el
    tracing de Langfuse se activa solo cuando `LANGFUSE_PUBLIC_KEY`/
    `LANGFUSE_SECRET_KEY` están configuradas (ver `.env.example`), así que el
    notebook funciona perfectamente sin ningún tracing configurado. Cuando
    están configuradas, cada llamada al agente de acá abajo queda registrada
    y es visible en la interfaz de Langfuse.
    """)
    return


@app.cell
def _(os):
    def get_callbacks() -> list:
        """Retorna los callbacks de LangChain para tracing (Langfuse si está configurado, si no, ninguno)."""
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
        """Construye un subagente scribe a partir de su prompt de tarea y su esquema de salida.

        Cada subagente comparte la misma receta de llamada al modelo, y solo
        difiere en su `prompt` de tarea, su `response_format`, las `tools`
        opcionales y (para diagnósticos con herramienta) el `call_limit` —
        es el mismo helper `build_agent` que `agent.py` usa para los cuatro
        nodos.

        `ModelCallLimitMiddleware` es una red de seguridad de fallo rápido:
        un agente de salida estructurada basado en herramientas que nunca
        llama a su herramienta oculta de estructuración (ver el nodo HPI más
        abajo) queda en bucle dentro de create_agent sin lanzar ninguna
        excepción, por lo que `ModelRetryMiddleware` por sí solo nunca lo
        detecta. Limitar el número de llamadas al modelo convierte ese bucle
        silencioso de varios minutos en un error claro.

        El valor por defecto de 8 le alcanza a los agentes de extracción de
        un solo paso. El agente de diagnósticos basado en herramientas
        (Parte 6) necesita un `call_limit` mucho más alto: el modelo pequeño
        vuelve a consultar la herramienta de ICD-10 de forma iterativa con
        nombres de diagnóstico reformulados en lugar de converger en una o
        dos llamadas, así que se llega a 8 antes de que produzca una
        respuesta final — ver Parte 6.
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
        """Invoca un subagente scribe sobre la transcripción y retorna su salida estructurada."""
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
    mo.md(f"**HPI generado:**\n\n{hpi_result.hpi}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parte 4 — Por qué cuatro subagentes enfocados, y no uno solo

    Sería más simple pedirle a un solo agente el `ClinicalNote` completo en
    una sola pasada. En la práctica, un modelo local pequeño al que se le
    pide completar un esquema grande mientras además corre una herramienta
    tiende a dejar campos vacíos — por ejemplo, deja de lado los signos
    vitales de forma consistente cuando también está ocupado validando
    códigos ICD-10 en la misma pasada. Por eso `agent.py` divide la nota en
    cuatro nodos enfocados — HPI, signos vitales, examen físico,
    diagnósticos — cada uno con un prompt acotado y un esquema acotado, y
    combina sus salidas después. Recién construimos el nodo de HPI arriba;
    signos vitales y examen físico siguen exactamente el mismo patrón.
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
    ## Parte 5 — Diagnósticos sin herramienta (la línea base)

    Antes de agregar la herramienta de ICD-10 en la Parte 6, veamos el agente
    de diagnósticos sin ella — es el comportamiento por defecto de `agent.py`
    (`--use-tool` es opcional). El modelo asigna códigos ICD-10-CM a partir de
    su propio conocimiento de entrenamiento, usando `prompts/diagnoses_prompt.txt`
    — la variante sin herramienta de las instrucciones de diagnósticos, en
    contraste con `diagnoses_prompt_with_tool.txt`.

    Incluso sin herramienta, este nodo sigue usando `ProviderStrategy(schema=...)`
    en vez de la estrategia por defecto basada en herramientas, el mismo
    razonamiento de JSON-schema nativo que el HPI en la Parte 3. La
    diferencia acá es el modelo: `agent.py` cambia a un modelo *sin
    razonamiento* (`reasoning=False`) para este camino. Con el razonamiento
    activado y sin ninguna herramienta que llamar, el modelo tiende a emitir
    un preámbulo largo de razonamiento libre antes de finalmente producir el
    JSON — lento, e innecesario cuando no hay ningún resultado de herramienta
    sobre el cual razonar.
    """)
    return


@app.cell
def _(ChatOllama, OLLAMA_MODEL, OLLAMA_TIMEOUT, httpx):
    # Se usa sin la herramienta de ICD-10, así que el razonamiento nunca se
    # activa cuando no hay ningún resultado de herramienta sobre el cual
    # razonar (contrastar con select_model en la Parte 6, que sí tiene
    # resultados de herramienta sobre los cuales razonar).
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
    ## Parte 6 — Diagnósticos, de forma determinista (extraer → buscar → seleccionar)

    La primera versión de este nodo le daba la herramienta de ICD-10
    directamente al agente de diagnósticos y dejaba que él decidiera, turno a
    turno, si buscar y cuántas veces hacerlo con `search_icd10_codes_batch`.
    En la práctica el modelo pequeño no converge: vuelve a consultar la
    herramienta una y otra vez con nombres de diagnóstico progresivamente
    reformulados en lugar de asentarse después de una sola llamada en lote.
    El flag `--deterministic` de `agent.py` reemplaza ese bucle autónomo por
    un pipeline fijo — esto es lo que hace.

    Tres pasos, de los cuales solo dos son llamadas al LLM — la llamada a la
    herramienta en sí nunca es una decisión del modelo:

    1. **Extraer candidatos** (LLM, sin herramientas) — listar cada
       diagnóstico candidato discutido o implícito en la transcripción, cada
       uno con un `search_term` para la búsqueda y un `candidate_name` con
       toda la especificidad para el eventual código. Todavía no se asigna
       ningún código.
    2. **Buscar una sola vez** (código plano, sin LLM) — deduplicar el
       `search_term` de cada candidato y llamar a `search_icd10_hybrid_batch`
       exactamente una vez, sin importar cuántos candidatos haya.
    3. **Seleccionar** (LLM, sin herramientas) — dados la transcripción y los
       resultados de búsqueda ya obtenidos para cada candidato, elegir el
       código correcto por candidato (u omitirlo si ninguno de sus
       resultados es una coincidencia genuina), y luego consolidar en
       diferenciales + valoración.

    Esto cambia la flexibilidad del agente basado en herramientas — podía
    volver a buscar si una primera pasada se veía débil — por una garantía
    sólida contra el modo de falla de reintentos/llamadas repetidas que lo
    motivó.
    """)
    return


@app.cell
def _(repo_root):
    icd10_path = str(repo_root / "data" / "ICD10_DB.parquet")
    return (icd10_path,)


@app.cell
def _(repo_root):
    diagnoses_candidates_prompt = (
        (repo_root / "prompts" / "diagnoses_candidates_prompt.txt").read_text(encoding="utf-8").strip()
    )
    return (diagnoses_candidates_prompt,)


@app.cell
async def _(
    ProviderStrategy,
    build_agent,
    diagnoses_candidates_prompt,
    extraction_model,
    models,
    run_agent,
    system_prompt,
    transcript,
):
    candidates_agent = build_agent(
        extraction_model,
        system_prompt,
        diagnoses_candidates_prompt,
        ProviderStrategy(schema=models.CandidateExtraction),
    )
    candidate_extraction = await run_agent(candidates_agent, transcript)
    candidate_extraction
    return candidate_extraction, candidates_agent


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **El paso 2 es código plano, no un agente.** El `search_term` de cada
    candidato se deduplica y se busca exactamente una vez — sin ningún LLM en
    el bucle, y sin manera de que el modelo decida volver a buscar.
    """)
    return


@app.cell
def _(candidate_extraction, icd10_path, search_icd10_hybrid_batch):
    search_terms = sorted({candidate.search_term for candidate in candidate_extraction.candidates})
    search_results = search_icd10_hybrid_batch(search_terms, limit=10, path=icd10_path)
    search_results
    return (search_results,)


@app.cell
def _(ChatOllama, OLLAMA_MODEL, OLLAMA_TIMEOUT, httpx):
    # Razonamiento activado, temp=0/con seed: probado en A/B contra un modelo
    # sin razonamiento sobre los mismos candidatos/search_results ya
    # obtenidos — sin razonamiento, el modelo conservaba de forma consistente
    # (incluso con temp=0) solo uno de varios candidatos bien emparejados,
    # descartando el resto por completo en vez de recorrer la lista
    # completa. num_predict es más alto para dejar espacio a ese
    # razonamiento.
    select_model = ChatOllama(
        model=OLLAMA_MODEL,
        num_ctx=20480,
        keep_alive="15m",
        validate_model_on_init=True,
        temperature=0.0,
        seed=42,
        reasoning=True,
        num_predict=8192,
        client_kwargs={
            "timeout": httpx.Timeout(connect=10.0, read=OLLAMA_TIMEOUT, write=30.0, pool=10.0)
        },
    )
    return (select_model,)


@app.cell
def _(repo_root):
    diagnoses_selection_prompt = (
        (repo_root / "prompts" / "diagnoses_selection_prompt.txt").read_text(encoding="utf-8").strip()
    )
    return (diagnoses_selection_prompt,)


@app.cell
async def _(
    ProviderStrategy,
    build_agent,
    candidate_extraction,
    diagnoses_selection_prompt,
    get_callbacks,
    json,
    models,
    search_results,
    select_model,
    system_prompt,
    transcript,
):
    # Incluir la transcripción original junto con los candidatos: sin ella,
    # este paso no tiene forma de notar que los resultados de búsqueda de un
    # candidato son todos malas coincidencias (en vez de opciones
    # razonables) y simplemente elige la menos mala en lugar de omitir el
    # candidato.
    candidate_blocks = [
        f"- candidate_name: {candidate.candidate_name}\n"
        f"  section: {candidate.section}\n"
        f"  search_results: {json.dumps(search_results.get(candidate.search_term, []))}"
        for candidate in candidate_extraction.candidates
    ]
    candidates_context = "\n".join(candidate_blocks) or "(no candidates extracted)"

    select_agent = build_agent(
        select_model,
        system_prompt,
        diagnoses_selection_prompt,
        ProviderStrategy(schema=models.DiagnosesOutput),
    )
    select_full_result = await select_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"<transcript>\n{transcript}\n</transcript>\n\n"
                        f"<candidates>\n{candidates_context}\n</candidates>"
                    ),
                }
            ]
        },
        config={"callbacks": get_callbacks()},
        stream=False,
    )
    diagnoses_result = select_full_result["structured_response"]

    # Reforzar en código que "cada código aparece en exactamente una lista",
    # en vez de confiar únicamente en la instrucción del prompt: el modelo
    # seguía duplicando un código en ambas listas de vez en cuando incluso
    # después de agregar esa regla al prompt.
    _assessment_codes = {d.icd10_code for d in diagnoses_result.assessment}
    diagnoses_result.differential_diagnoses = [
        d for d in diagnoses_result.differential_diagnoses if d.icd10_code not in _assessment_codes
    ]
    diagnoses_result
    return (select_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parte 7 — Componiendo los nodos en un grafo

    Cuatro subagentes independientes todavía no son un pipeline de agente.
    `agent.py` los conecta en un `StateGraph` de LangGraph: un estado
    tipado (`ScribeState`) del que cada nodo lee `transcript` y en el que
    escribe un campo, ejecutados de forma secuencial (una llamada a Ollama a
    la vez — confiable contra un único servidor local).

    Reutilizamos exactamente las piezas construidas arriba: `hpi_agent`,
    `vitals_agent` y `physical_exam_agent` de las Partes 3–4, más el pipeline
    determinista de extraer/buscar/seleccionar diagnósticos de la Parte 6 (no
    las líneas base basada en herramientas o sin herramienta) — componerlos
    es solo cableado, sin lógica de extracción nueva. El nodo de diagnósticos
    es la única excepción a "una llamada a Ollama por nodo": en realidad son
    tres pasos (dos llamadas al LLM alrededor de una búsqueda de ICD-10
    plana), envueltos en una sola función para que siga viéndose como
    cualquier otro nodo desde el punto de vista del grafo.
    """)
    return


@app.cell
def _(TypedDict, models):
    class ScribeState(TypedDict):
        """Estado que atraviesa el grafo scribe: una entrada, cuatro salidas."""

        transcript: str
        hpi: models.HistoryOfPresentIllness
        vitals: models.VitalSigns
        physical_exam: models.PhysicalExam
        diagnoses: models.DiagnosesOutput

    return (ScribeState,)


@app.cell
def _(
    candidates_agent,
    get_callbacks,
    hpi_agent,
    icd10_path,
    json,
    physical_exam_agent,
    run_agent,
    search_icd10_hybrid_batch,
    select_agent,
    vitals_agent,
):
    async def hpi_node(state):
        return {"hpi": await run_agent(hpi_agent, state["transcript"])}

    async def vitals_node(state):
        return {"vitals": await run_agent(vitals_agent, state["transcript"])}

    async def physical_exam_node(state):
        return {"physical_exam": await run_agent(physical_exam_agent, state["transcript"])}

    async def diagnoses_node(state):
        # Refleja los tres pasos de la Parte 6, corridos contra la
        # transcripción propia de este nodo en vez de la variable
        # `transcript` a nivel del notebook.
        transcript = state["transcript"]
        candidate_extraction = await run_agent(candidates_agent, transcript)

        search_terms = sorted({c.search_term for c in candidate_extraction.candidates})
        search_results = search_icd10_hybrid_batch(search_terms, limit=10, path=icd10_path)

        candidate_blocks = [
            f"- candidate_name: {c.candidate_name}\n"
            f"  section: {c.section}\n"
            f"  search_results: {json.dumps(search_results.get(c.search_term, []))}"
            for c in candidate_extraction.candidates
        ]
        candidates_context = "\n".join(candidate_blocks) or "(no candidates extracted)"

        select_full_result = await select_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"<transcript>\n{transcript}\n</transcript>\n\n"
                            f"<candidates>\n{candidates_context}\n</candidates>"
                        ),
                    }
                ]
            },
            config={"callbacks": get_callbacks()},
            stream=False,
        )
        diagnoses = select_full_result["structured_response"]

        assessment_codes = {d.icd10_code for d in diagnoses.assessment}
        diagnoses.differential_diagnoses = [
            d for d in diagnoses.differential_diagnoses if d.icd10_code not in assessment_codes
        ]
        return {"diagnoses": diagnoses}

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
    ## Parte 8 — El `ClinicalNote` ensamblado

    `extract_note()` en `agent.py` hace exactamente esta combinación: extrae
    el resultado de cada nodo desde el estado final del grafo y ensambla un
    único `ClinicalNote`.
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
    # Enviar cualquier trace pendiente de Langfuse antes de seguir, igual que
    # main() en agent.py — si no, los spans acumulados durante esta sesión
    # podrían no llegar a enviarse.
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse import get_client

        get_client().flush()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sigue: notebook 2 — evaluando lo que produce el agente

    Este notebook solo construye; nunca pregunta si la salida es *buena*. Eso
    es trabajo de `evals/`: chequeos deterministas (`eval_diagnoses.py`,
    `eval_vitals.py`, `eval_trajectory.py`) que comparan la salida extraída
    contra `data/golden/golden_encounter_*.json`, más scoring con
    LLM-as-a-Judge (`evals/eval_hpi_judge.py`, `evals/eval_clinical_note_dataset.py`)
    para las secciones de texto libre que un diff determinista no puede
    calificar. Ver la sección "Evals" del README para saber cómo correr cada
    uno — o ejecutar la CLI directamente sobre esta misma transcripción:

    ```
    uv run python agent.py data/encounter_riv001.txt --use-tool -o outputs/encounter_riv001.json
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
