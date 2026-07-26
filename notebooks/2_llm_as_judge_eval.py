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
    # Evaluando un Agente Scribe Médico con LLM-as-a-Judge

    Notebook 2 de la serie del taller. El notebook 1 construyó el agente;
    este notebook pregunta si lo que produce es *bueno* — específicamente
    para las dos secciones que un diff determinista no puede calificar: la
    narrativa del HPI y los hallazgos del examen físico (PE). Los
    diagnósticos (códigos ICD-10) y los signos vitales (campos numéricos
    fijos) reciben evals de coincidencia exacta/tolerancia en cambio
    (`evals/eval_diagnoses.py`, `evals/eval_vitals.py`) — fuera del alcance
    acá.

    Reconstruimos los nodos `hpi`/`physical_exam` exactamente como lo hizo el
    notebook 1 (mismo razonamiento, no repetido acá — ver el notebook 1 para
    entender por qué `ProviderStrategy`, nodos sin herramientas, etc.), y
    después construimos en vivo la maquinaria del judge: los mismos prompts,
    esquemas y el helper `invoke_judge` que usan en producción
    `evals/eval_hpi_judge.py` y `evals/eval_clinical_note_dataset.py`,
    importados directamente cada vez que hacerlo es seguro (ver la Parte 4).
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
    for _p in (repo_root, repo_root / "evals"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    load_dotenv(repo_root / ".env")
    return os, repo_root


@app.cell
def _():
    import httpx
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
    from langchain.agents.structured_output import ProviderStrategy
    from langchain_ollama import ChatOllama

    return (
        ChatOllama,
        ModelCallLimitMiddleware,
        ModelRetryMiddleware,
        ProviderStrategy,
        create_agent,
        httpx,
    )


@app.cell
def _():
    import models

    return (models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parte 1 — Regenerar los mismos dos nodos que el notebook 1

    Misma transcripción (`RIV-001`), misma receta `build_agent`/`run_agent`,
    mismo `extraction_model` — condensado acá porque el notebook 1 ya cubre
    el "por qué" de cada decisión (sin herramientas, `ProviderStrategy(schema=...)`
    para el campo de texto libre del HPI, temperature 0 para una extracción
    casi determinista).
    """)
    return


@app.cell
def _(repo_root):
    transcript_path = repo_root / "data" / "encounter_riv001.txt"
    transcript = transcript_path.read_text(encoding="utf-8")
    return (transcript,)


@app.cell
def _(repo_root):
    system_prompt = (repo_root / "prompts" / "system_prompt.txt").read_text(encoding="utf-8").strip()
    return (system_prompt,)


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
def _(ChatOllama, httpx, os):
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))

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
    return OLLAMA_MODEL, extraction_model


@app.cell
def _(
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    create_agent,
    get_callbacks,
):
    def build_agent(llm, system_prompt, task_prompt, response_format, tools=None, call_limit=8):
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
async def _(
    ProviderStrategy,
    build_agent,
    extraction_model,
    models,
    repo_root,
    run_agent,
    system_prompt,
    transcript,
):
    hpi_prompt = (repo_root / "prompts" / "hpi_prompt.txt").read_text(encoding="utf-8").strip()
    hpi_agent = build_agent(
        extraction_model,
        system_prompt,
        hpi_prompt,
        ProviderStrategy(schema=models.HistoryOfPresentIllness),
    )
    hpi_result = await run_agent(hpi_agent, transcript)
    hpi_result
    return (hpi_result,)


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
    physical_exam_agent = build_agent(extraction_model, system_prompt, physical_exam_prompt, models.PhysicalExam)
    physical_exam_result = await run_agent(physical_exam_agent, transcript)
    physical_exam_result
    return (physical_exam_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parte 2 — Por qué estas dos secciones necesitan un judge, no un diff

    `evals/eval_diagnoses.py` y `evals/eval_vitals.py` pueden comparar sus
    salidas contra un registro de referencia campo por campo: un código
    ICD-10 coincide o no coincide, una frecuencia cardíaca está dentro de
    tolerancia o no lo está. La narrativa del HPI y los hallazgos del PE son
    texto libre — no existe una única versión correcta contra la cual hacer
    diff, así que calificarlos significa pedirle a un segundo modelo que lea
    la transcripción y la salida y la juzgue, de la misma manera en que lo
    haría un revisor humano.

    Tres decisiones de diseño heredadas de `evals/eval_hpi_judge.py`,
    `evals/eval_clinical_note_dataset.py` y las notas de diseño de `eval.md`:

    - **Descomponer en dimensiones evaluables de forma independiente**
      (Accuracy, Completeness, Tone) en vez de un solo score general — un
      único número no puede decirte *qué* está mal, y un criterio amplio es
      difícil de aplicar de forma consistente para un judge.
    - **Scoring pointwise** (calificar una salida contra una rúbrica) en vez
      de una comparación pairwise/listwise — acá no hay un segundo
      candidato contra el cual comparar, solo la transcripción y la salida.
    - **Anclas de comportamiento concretas por cada punto del score**, no una
      escala simple de 0 a 4 — ver las rúbricas en `prompts/hpi_judge_prompt.txt`
      y `prompts/physical_exam_judge_prompt.txt`, cada nivel de score atado a
      una condición específica y verificable en vez de un adjetivo vago.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parte 3 — El contrato del judge: score *y* justificación, por dimensión

    `HPIJudgeScore`/`PEJudgeScore` en `models.py` acoplan cada campo
    `*_score` (0-4) con un string `*_rationale`. La justificación no es
    decorativa: un judge que solo emite un número no te da forma de saber si
    realmente está leyendo la transcripción o solo reconociendo patrones
    superficiales — la explicación es justamente lo que expondría un sesgo de
    posición, un sesgo de verbosidad o un sesgo de autofavorecimiento si
    alguno estuviera presente.
    """)
    return


@app.cell
def _(mo, models):
    mo.accordion(
        {
            "HPIJudgeScore": mo.json(models.HPIJudgeScore.model_json_schema()),
            "PEJudgeScore": mo.json(models.PEJudgeScore.model_json_schema()),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parte 4 — Selección del modelo judge: nunca el mismo generador

    `invoke_judge` en `evals/judge_client.py` es la única función que llaman
    ambos scripts de eval en producción — importada directamente acá en vez
    de reconstruida, ya que no tiene efectos secundarios al importarse (a
    diferencia de `agent.py`, que hace ping a Ollama apenas se importa — ver
    el notebook 1). Hace scoring con Amazon Bedrock por defecto, y cae a un
    modelo local de Ollama si Bedrock no está disponible — deliberadamente
    **no** `OLLAMA_MODEL` (el generador propio de la app), así que judge !=
    generador se mantiene incluso durante una caída de Bedrock. Un judge que
    comparte los pesos del generador tiende a calificar su propia salida de
    forma más favorable (sesgo de autofavorecimiento), que es exactamente lo
    que esto evita.
    """)
    return


@app.cell
def _():
    import judge_client

    return (judge_client,)


@app.cell
def _(OLLAMA_MODEL, judge_client, mo):
    mo.md(
        f"**Judge primario:** Bedrock `{judge_client.BEDROCK_MODEL_ID}` "
        f"(región `{judge_client.BEDROCK_REGION}`)\n\n"
        f"**Judge de fallback:** Ollama `{judge_client.FALLBACK_OLLAMA_MODEL}` — "
        f"distinto del generador de este notebook, Ollama `{OLLAMA_MODEL}`, así "
        f"que judge != generador se mantiene incluso en el camino de fallback."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parte 5 — Evaluando el HPI

    `evals/eval_hpi_judge.py` no tiene ningún `import agent` ni ningún otro
    código a nivel de módulo con efectos secundarios, así que su ruta de
    prompt, sus definiciones de score config y su función `run_judge` se
    importan directamente en vez de copiarse — este notebook y el script de
    producción comparten exactamente la misma lógica de evaluación, no una
    parecida.
    """)
    return


@app.cell
def _():
    from eval_hpi_judge import JUDGE_PROMPT_PATH as HPI_JUDGE_PROMPT_PATH
    from eval_hpi_judge import SCORE_CONFIGS as HPI_SCORE_CONFIGS
    from eval_hpi_judge import run_judge as run_hpi_judge

    return HPI_JUDGE_PROMPT_PATH, run_hpi_judge


@app.cell
def _(HPI_JUDGE_PROMPT_PATH, os, repo_root):
    os.environ["AWS_PROFILE"] = os.getenv("AWS_PROFILE", "default")
    hpi_judge_prompt = (repo_root / HPI_JUDGE_PROMPT_PATH).read_text(encoding="utf-8").strip()
    return (hpi_judge_prompt,)


@app.cell
def _(hpi_judge_prompt, hpi_result, run_hpi_judge, transcript):
    hpi_score, hpi_judge_model = run_hpi_judge(hpi_judge_prompt, transcript, hpi_result.hpi)
    hpi_judge_model, hpi_score
    return (hpi_score,)


@app.cell
def _(hpi_score, mo):
    mo.md(f"""
    | Dimensión | Score | Justificación |
    |---|---|---|
    | Accuracy | {hpi_score.accuracy_score}/4 | {hpi_score.accuracy_rationale} |
    | Completeness | {hpi_score.completeness_score}/4 | {hpi_score.completeness_rationale} |
    | Tone | {hpi_score.tone_score}/4 | {hpi_score.tone_rationale} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ¿La rúbrica realmente discrimina?

    Un judge que le pone 4/4 a cualquier HPI sin importar el contenido no
    sirve para nada. Como una prueba rápida de calibración (la misma idea que
    la guía de Hamel Husain de validar una rúbrica contra ejemplos
    deliberadamente malos), truncamos el HPI generado hasta su primera
    oración — como si fuera un modelo que dejó de lado la mayor parte del
    encuentro — y evaluamos eso en cambio. Completeness debería caer
    fuertemente; Accuracy debería mantenerse alto (lo que queda sigue siendo
    verdadero, solo que incompleto).
    """)
    return


@app.cell
def _(hpi_judge_prompt, hpi_result, run_hpi_judge, transcript):
    degraded_hpi = hpi_result.hpi.split(". ")[0].strip() + "."
    degraded_hpi_score, degraded_hpi_judge_model = run_hpi_judge(hpi_judge_prompt, transcript, degraded_hpi)
    degraded_hpi, degraded_hpi_judge_model, degraded_hpi_score
    return (degraded_hpi_score,)


@app.cell
def _(degraded_hpi_score, hpi_score, mo):
    mo.md(f"""
    | Dimensión | HPI completo | HPI truncado |
    |---|---|---|
    | Accuracy | {hpi_score.accuracy_score}/4 | {degraded_hpi_score.accuracy_score}/4 |
    | Completeness | {hpi_score.completeness_score}/4 | {degraded_hpi_score.completeness_score}/4 |
    | Tone | {hpi_score.tone_score}/4 | {degraded_hpi_score.tone_score}/4 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parte 6 — Evaluando el examen físico (PE)

    No existe un `eval_physical_exam_judge.py` independiente: la llamada al
    judge de PE vive de forma inline como `pe_judge_evaluator` dentro de
    `evals/eval_clinical_note_dataset.py`, que no se puede importar
    directamente acá — importa `agent.py` a nivel de módulo para su función
    de tarea, exactamente el efecto secundario que el notebook 1 evita. Todo
    lo reutilizable *sin* ese import — `models.PEJudgeScore`,
    `prompts/physical_exam_judge_prompt.txt`, y `judge_client.invoke_judge`
    (ya importado en la Parte 4) — se reutiliza directamente; solo el
    pequeño pegamento de formateo de mensajes entre ellos se reconstruye acá,
    replicando el mismo formato que usa `pe_judge_evaluator`
    (`"- {system}: {findings}"` por entrada).
    """)
    return


@app.cell
def _(repo_root):
    pe_judge_prompt = (
        (repo_root / "prompts" / "physical_exam_judge_prompt.txt").read_text(encoding="utf-8").strip()
    )
    return (pe_judge_prompt,)


@app.cell
def _(judge_client, models):
    def run_pe_judge(judge_prompt, transcript, findings):
        pe_text = "\n".join(f"- {f.system.value}: {f.findings}" for f in findings)
        messages = [
            {"role": "system", "content": judge_prompt},
            {
                "role": "user",
                "content": f"<transcript>\n{transcript}\n</transcript>\n\n<physical_exam>\n{pe_text}\n</physical_exam>",
            },
        ]
        return judge_client.invoke_judge(models.PEJudgeScore, messages)

    return (run_pe_judge,)


@app.cell
def _(pe_judge_prompt, physical_exam_result, run_pe_judge, transcript):
    pe_score, pe_judge_model = run_pe_judge(pe_judge_prompt, transcript, physical_exam_result.findings)
    pe_judge_model, pe_score
    return (pe_score,)


@app.cell
def _(mo, pe_score):
    mo.md(f"""
    | Dimensión | Score | Justificación |
    |---|---|---|
    | Accuracy | {pe_score.accuracy_score}/4 | {pe_score.accuracy_rationale} |
    | Completeness | {pe_score.completeness_score}/4 | {pe_score.completeness_rationale} |
    | Tone | {pe_score.tone_score}/4 | {pe_score.tone_rationale} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prueba de calibración: un hallazgo autorreportado mal clasificado como hallazgo de examen

    `prompts/physical_exam_judge_prompt.txt` lo menciona específicamente: "un
    hallazgo construido enteramente a partir del reporte propio del paciente,
    en vez de una observación/medición del examinador, es en sí mismo un
    problema de accuracy". Reescribimos un hallazgo con las palabras propias
    del paciente y volvemos a evaluar — Accuracy (y probablemente Tone)
    debería caer en esta variante en relación con el original.
    """)
    return


@app.cell
def _(physical_exam_result):
    _findings = list(physical_exam_result.findings)
    degraded_pe_findings = [
        _findings[0].model_copy(update={"findings": "Patient says the wound feels cold and really hurts."}),
        *_findings[1:],
    ]
    degraded_pe_findings
    return (degraded_pe_findings,)


@app.cell
def _(degraded_pe_findings, pe_judge_prompt, run_pe_judge, transcript):
    degraded_pe_score, degraded_pe_judge_model = run_pe_judge(pe_judge_prompt, transcript, degraded_pe_findings)
    degraded_pe_judge_model, degraded_pe_score
    return (degraded_pe_score,)


@app.cell
def _(degraded_pe_score, mo, pe_score):
    mo.md(f"""
    | Dimensión | Hallazgos originales | Hallazgo autorreportado incorporado |
    |---|---|---|
    | Accuracy | {pe_score.accuracy_score}/4 | {degraded_pe_score.accuracy_score}/4 |
    | Completeness | {pe_score.completeness_score}/4 | {degraded_pe_score.completeness_score}/4 |
    | Tone | {pe_score.tone_score}/4 | {degraded_pe_score.tone_score}/4 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parte 7 — Lo que el judge deliberadamente *no* evalúa

    `PEJudgeScore.completeness_score` (ver el docstring de `models.py`) está
    acotado a la profundidad *dentro de los sistemas que el modelo ya
    incluyó* — nunca penaliza que falte un sistema corporal por completo.
    `evals/eval_clinical_note_dataset.py` evalúa esa pregunta de cobertura
    por separado, de forma determinista, como precision/recall contra un
    conjunto de referencia de sistemas (`pe_precision_recall_evaluator`),
    filtrando primero las entradas placeholder repetitivas de "no se
    examinó" (`_is_filler`) para que rellenar cada sistema con una nota
    genérica no infle la precision. Dividir la métrica de esta manera
    significa que un Completeness bajo siempre quiere decir "poco profundo en
    un sistema incluido", nunca "falta un sistema" — son modos de falla
    distintos con soluciones distintas.

    Reconstruido acá abajo para `RIV-001` contra la misma referencia armada a
    mano que usa `evals/eval_clinical_note_dataset.py` para ese encuentro (su
    `PE_SYSTEM_REFERENCE["RIV-001"]` — el propio campo `physical_exam` del
    JSON de referencia es prosa de texto libre, no un desglose por sistema,
    así que no puede alimentar esta métrica directamente; ver el docstring de
    ese script).
    """)
    return


@app.cell
def _():
    import re

    # Refleja `_is_filler` en evals/eval_clinical_note_dataset.py: un sistema
    # predicho solo cuenta como verdadero positivo si sus hallazgos son algo
    # más que un placeholder repetitivo de "no se examinó".
    _filler_patterns = [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"no specific .*(findings|exam)",
            r"no .*(examination|exam|assessment) (was )?performed",
            r"no .*(findings|complaints) (noted|documented|stated)",
            r"not (examined|assessed|performed|documented)",
        ]
    ]

    def is_filler(findings_text: str) -> bool:
        text = (findings_text or "").strip()
        if not text:
            return True
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return all(any(p.search(s) for p in _filler_patterns) for s in sentences)

    return (is_filler,)


@app.cell
def _():
    golden_pe_systems_riv001 = {"general", "skin", "neurologic"}
    return (golden_pe_systems_riv001,)


@app.cell
def _(golden_pe_systems_riv001, is_filler, physical_exam_result):
    predicted_pe_systems = {
        f.system.value for f in physical_exam_result.findings if not is_filler(f.findings)
    }
    _true_positives = predicted_pe_systems & golden_pe_systems_riv001
    pe_precision = len(_true_positives) / len(predicted_pe_systems) if predicted_pe_systems else 0.0
    pe_recall = len(_true_positives) / len(golden_pe_systems_riv001) if golden_pe_systems_riv001 else 0.0
    predicted_pe_systems, pe_precision, pe_recall
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sigue: registrar estos scores donde realmente son útiles

    Todo lo de arriba corrió en memoria y se imprimió en este notebook. En
    producción, `evals/eval_hpi_judge.py` adjunta sus scores a la
    observation `hpi` de un trace en vivo de Langfuse (`client.create_score(...)`),
    y `evals/eval_clinical_note_dataset.py` adjunta las ocho métricas — las
    tres del HPI, las tres del PE, más `pe_precision`/`pe_recall` — a una
    Experiment run de Langfuse sobre el dataset de referencia, así que cada
    métrica de ambas secciones aparece lado a lado en la interfaz a través de
    cambios de modelo/prompt. Ver esos scripts (y la sección "Evals" del
    README) para ese cableado; este notebook se detiene en el paso de
    evaluación en sí.
    """)
    return


@app.cell
def _(os):
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse import get_client

        get_client().flush()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
