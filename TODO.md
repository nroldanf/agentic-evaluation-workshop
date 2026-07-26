TODO:
- Pensar en qué forma podríamos orientar lo que ya tenemos para hacerlo más dinámico a los atendees del workshop e.g. hacer que escriban una función, o que repitan un proceso como los evals de diagnostico pero con vitals y visualizen en langfuse [MAFECITA]
- Mover el compose de langfuse al repositorio, e incluir instrucciones de cómo levantarlo localmente para que los asistentes puedan ver sus propios resultados [MAFECITA] ✅
- Definir e implementar los evals para physical exam [NICOLAS] ✅
- Visualizar los resultados de los evals en langfuse [NICOLAS Y MAFECITA]
- Traducir los scripts a ingles para tener todo consistente [MAFECITA] ✅
- Poner en la presentación las escenas donde suceden las lesiones [NICOLAS]
  - Introducir al señor de los anillos
  - Plantear un escenario what if? (what if... existiera magia que permitiese medir cosas como los signos vitales como lo hacemos hoy en día, y que los elfos tuvieran asistentes? - en este caso los agentes?)
  - Espada del Morgul: https://www.youtube.com/watch?v=NCjDuTr4BjA
  - Amputación del dedo: https://www.youtube.com/watch?v=c24-0Amwyik
- Añadir valores de vitals de nuevo a los transcript [MAFECITA]
- Incluir el QR para calificar la presentación: https://drive.google.com/file/d/1n6d6kv7NiC4Quk-86xau-RHYGiBihRqU/view ✅
- Remover codigo no necesario para el workshop ✅
  - Graph paralelo
  - cache (optional)
- Actualizar la diapositiva con el diagrama para incluir vitals [NICOLAS] ✅
- Generar un readme en español y otro en ingles [NICOLAS] ✅
- Borrar skills y cosas de claude code [NICOLAS] ✅
- Añadir como opción el uso de API como google o chatgpt [NICOLAS]
- Crear una pregunta de evaluación [MAFECITA]
- Unir trajectory evals con la salida de la agente [NICOLAS]

# HPI LLM-as-a-Judge Eval — Design Plan

Design for an LLM-as-a-Judge evaluator that grades the `hpi` field produced by
the `hpi` node (`prompts/hpi_prompt.txt` → `HistoryOfPresentIllness.hpi`)
against the source transcript, on three custom dimensions — **Accuracy**,
**Completeness**, **Tone** — each scored **0–4**. This is a design doc only;
nothing here is implemented yet.

## 1. Scope

Target field: `hpi: str` — a single chronological narrative paragraph
(`models.py:67-77`). Input available to the judge: the full `data/transcript.txt`
patient–doctor dialogue. No separate "expected HPI" exists today — the judge
grades the HPI **against the transcript**, not against a reference summary
(see §7 for why we may still want an optional gold reference).

## 2. Grounding sources and what each contributes

| Source | Core takeaway used here |
|---|---|
| [Hamel Husain — Your AI Product Needs Evals](https://hamel.dev/blog/posts/llm-judge/) | Multi-point scales are only actionable if every point has a *concrete behavioral anchor*; otherwise raters can't tell a 3 from a 4. Calibrate the judge against ~30+ human-labeled examples, growing until no new failure modes appear. Use **critique shadowing**: humans write detailed pass/fail rationales, those become few-shot examples for the judge. Track agreement with class-imbalance-aware metrics (precision/recall), not raw agreement. Prefer deterministic code checks for anything mechanical/structural. |
| [Loka — Can AI Make Good Decisions About Itself?](https://www.loka.com/blog/can-ai-make-good-decisions-about-itself) | Judge design = 4 inputs (eval type, criteria, item, reference) → 3 outputs (score, explanation, feedback). Decompose broad criteria into independently assessable dimensions (exactly what we're doing with Accuracy/Completeness/Tone). Prefer **pointwise** scoring (score one output against criteria) over pairwise/listwise when there's no second candidate to compare against — our case. |
| [Loka — Can AI Make Good Decisions About Itself? (Part 2)](https://www.loka.com/blog/can-ai-make-good-decisions-about-itself-2) | GPT-4-class judges reach ~85% agreement with humans (human-human agreement was ~81%), but exhibit **position bias**, **verbosity bias**, **self-enhancement bias** (~25% self-score inflation), and measurable **diversity/demographic bias** in clinical recommendations. Explanations (not just scores) are what expose these biases. Recommended pattern: LLM scores first, human reviews/refines, both versions kept. |
| [Langfuse docs — LLM-as-a-Judge, Scores, Evaluation Core Concepts](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge) | Concrete mechanics of how a "custom evaluator" is authored in Langfuse (rubric prompt + `{{variables}}` + score type), how it's wired to a target (observation vs. trace vs. experiment), and how scores/annotations/experiments compose. Used throughout §6–§9 below. |

### Tensions this plan resolves

- **The user wants a 0–4 numeric scale; Hamel argues against unanchored
  Likert scales.** We keep 0–4 (explicit requirement) but make every point a
  distinguishable, HPI-specific behavior (§4) rather than "somewhat
  accurate." We also validate the scale with ordinal-aware agreement
  metrics (quadratic-weighted κ / Spearman, plus within-1 agreement), and
  optionally collapse to a ≥3 "pass" gate for monitoring, where
  precision/recall applies per Hamel's imbalance warning (§8).
- **Accuracy/Completeness vs. Tone use different reference bases.** Accuracy
  and Completeness are graded *against the transcript* (faithfulness /
  coverage). Tone is graded *against physician-documentation norms*, **not**
  the transcript — the transcript is casual spoken dialogue; a good HPI
  transforms it into clinical prose, so scoring Tone by literal similarity to
  the transcript would penalize exactly the transformation we want. The three
  judge prompts must not share a single "compare to transcript" framing.

## 3. Resolved: HPI scope includes meds/allergies/social history

**Decision:** medications, allergies, and smoking/social history are
**in-scope** for the HPI narrative. This matches current agent behavior
(`outputs/clinical_note_new.json` already folds lisinopril, the penicillin
allergy, and smoking history into the HPI) and, more importantly, the
`ClinicalNote` schema (`models.py`) has no separate fields for
medications/allergies/social history today — marking them out-of-scope for
HPI would mean that information is captured nowhere in the structured note.

Consequence for the rubrics below: **Completeness** expects meds/allergies/
smoking history to appear in the HPI when the transcript mentions them
(omitting them costs points); **Tone** never penalizes their inclusion.
Revisit this if a dedicated Medications/Allergies/Social-History field is
ever added to `ClinicalNote` — at that point HPI scope should likely narrow
back to the present-illness narrative only, and this decision should be
re-made.

## 4. Dimension rubrics (0–4, behaviorally anchored)

### Accuracy — faithfulness to the transcript

| Score | Anchor |
|---|---|
| 4 | Every claim in the HPI is directly supported by the transcript. Correct chronology, correct polarity (present vs. denied), correct values/timing. No fabrication. |
| 3 | One minor wording deviation that does not change clinical meaning (e.g., paraphrasing "come and gone" as "resolved" when the transcript implies intermittent, not resolved — a subtle but real drift to flag as an example, not necessarily always a 3). |
| 2 | One clinically relevant inaccuracy (wrong severity, wrong duration, wrong timing, incorrect associated symptom) that would change a clinician's read, OR several minor issues stacked together. |
| 1 | Multiple clinically relevant inaccuracies, or one fabricated symptom/finding not present in the transcript at all. |
| 0 | Systematically contradicts the transcript, or fabricates the chief complaint or a major element of the history. |

### Completeness — coverage of transcript content relevant to present illness

| Score | Anchor |
|---|---|
| 4 | Captures chief complaint, onset, duration, chronological progression, all associated/prodromal symptoms, exacerbating/alleviating factors, treatments attempted, and any medications/allergies/smoking history the patient mentioned — everything the transcript states that's relevant to the encounter. |
| 3 | Omits one non-critical detail (e.g., one minor associated symptom) that doesn't change the clinical picture. |
| 2 | Missing one clinically relevant element a clinician would expect (e.g., a pertinent negative, an exacerbating factor, a treatment attempted). |
| 1 | Missing multiple clinically relevant elements; the HPI reads as a fragment rather than a full narrative of the encounter. |
| 0 | Severely incomplete — captures only the chief complaint, or omits the narrative almost entirely. |

### Tone — physician-documentation register (graded against clinical-note norms, NOT the transcript)

| Score | Anchor |
|---|---|
| 4 | Fully professional third-person clinical prose; standard phrasing (e.g., "reports," "denies," "notes"); single flowing paragraph; no verbatim patient quotes or colloquialisms; follows the mandated opening template exactly (`prompts/hpi_prompt.txt:3-6`). |
| 3 | Minor style lapse (one stray informal phrase) that a light copy-edit would fix; still clearly reads as a clinical note. |
| 2 | Noticeable tone issues: mixes in patient's own words/slang, awkward phrasing, or doesn't follow the mandated opening — but is still recognizably an attempt at clinical documentation. |
| 1 | Reads more like a conversational recap of the transcript than a clinical note; frequent first/second-person artifacts or informal language. |
| 0 | Unprofessional or inappropriate register; not usable as clinical documentation without a full rewrite. |

**Note:** the mandated opening line and single-paragraph format
(`prompts/hpi_prompt.txt`) are deterministic, checkable rules. Per Hamel's
guidance to prefer code assertions for mechanical checks, these are better
served by a Langfuse **code evaluator** (regex/structural lint) run alongside
the LLM judge, rather than folded into the Tone score or added as a 4th
dimension — the user asked for exactly three dimensions; keep it that way and
treat structural linting as a complementary, separate signal.

## 5. Judge prompt design

One prompt per dimension (see §6 for why — Langfuse's custom evaluator model
is one numeric score per evaluator). Shared shape:

```
ROLE: You are an experienced attending physician reviewing a colleague's
clinical documentation for quality assurance.

TASK: Score the HPI below on <DIMENSION> using the rubric. <DIMENSION-
SPECIFIC REFERENCE INSTRUCTION — "against the transcript" for Accuracy/
Completeness, "against standard physician-documentation conventions" for
Tone>.

RUBRIC:
<the 0-4 anchor table for this dimension, verbatim>

FEW-SHOT EXAMPLES:
<2-4 worked examples with transcript excerpt, HPI excerpt, correct score,
and a human-written rationale — sourced from the calibration set, §8>

TRANSCRIPT:
{{transcript}}

HPI TO GRADE:
{{hpi}}

OUTPUT: a JSON object with:
  - score: integer 0-4
  - rationale: 2-4 sentences citing the specific transcript/HPI evidence
    that drove the score
```

The `rationale` is not cosmetic: per Loka pt. 2, explanations are the
mechanism that surfaces hidden bias (verbosity reward, demographic skew,
self-preference) that a bare number hides. In Langfuse this rationale is
stored in `Score.comment`.

## 6. Mapping onto Langfuse

### 6.1 Score Configs (one per dimension)

Per [Scores Data Model](https://langfuse.com/docs/evaluation/scores/data-model),
a `ScoreConfig` pins name, data type, and range so scores are comparable
across runs/team members:

| ScoreConfig name | dataType | minValue | maxValue | description |
|---|---|---|---|---|
| `hpi_accuracy` | NUMERIC | 0 | 4 | Faithfulness of the HPI to the transcript — see rubric §4. |
| `hpi_completeness` | NUMERIC | 0 | 4 | Coverage of transcript content relevant to the present illness — see rubric §4. |
| `hpi_tone` | NUMERIC | 0 | 4 | Physician-documentation register — see rubric §4. |

### 6.2 Evaluators: one per dimension, not one evaluator emitting three scores

Langfuse's custom LLM-as-a-Judge evaluator is authored as **one rubric prompt
→ one score type** (numeric, categorical, or boolean); categorical evaluators
can emit multiple scores only across *categories* of the same evaluator, not
across unrelated dimensions. So the native mapping is **three evaluators**:
`hpi-accuracy-judge`, `hpi-completeness-judge`, `hpi-tone-judge`, each tied to
its ScoreConfig above, each with its own rubric prompt from §5.

This is the right choice for the **online/production lane** (§6.3) because it
needs no code — configured entirely in the Evaluators UI (or the unstable
Evaluators/Evaluation Rules API for version control). Per docs, this also
gives per-dimension debug traces (filterable by the
`langfuse-llm-as-a-judge` environment) and lets each dimension be
independently sampled/filtered ("compositional evaluation").

**Alternative for the offline/experiment lane** (§6.4, where we're already
writing code to run the pipeline): a single SDK-side judge call that reads
transcript + HPI once and emits three `langfuse.score()` calls
(`hpi_accuracy`, `hpi_completeness`, `hpi_tone` against the matching
ScoreConfigs). This is ~3x cheaper per item (one LLM call vs. three) and
keeps the three scores internally consistent (same judge context/pass).
Recommend it for regression runs over a dataset; keep the native 3-evaluator
approach for zero-code production monitoring.

### 6.3 Online lane — observation-level evaluators (recommended over trace-level)

Docs are explicit that trace-level LLM-as-a-Judge is **legacy**;
observation-level evaluators complete in seconds vs. minutes and give
operation-level precision. Plan:

- **Target**: the observation produced by the `hpi` LangGraph node
  (`agent.py:319-323`), which is auto-instrumented via
  `langfuse.langchain.CallbackHandler` (`agent.py:59-73`) — no manual
  `@observe` calls exist today.
- **Before finalizing the mapping, inspect one live trace.** The `hpi` node
  wraps an agent call (`run_agent(agent, state["transcript"], config)`); the
  *node span*'s input is plausibly the plain transcript string, while the
  nested *LLM generation* underneath it has input as a full
  system+human message array. Confirm which observation actually carries
  `input = transcript` and `output` containing the `hpi` field (likely via a
  JSONPath like `$.hpi` if the output is the full `HistoryOfPresentIllness`
  object) before wiring `{{transcript}}`/`{{hpi}}` mappings — do not assume.
- **Filter**: by observation name (`hpi`, once confirmed) rather than by
  trace-level tags, to avoid needing `propagate_attributes()` (only required
  if filtering by trace-level attrs like tags/userId on an observation-level
  rule — not needed for a name-based filter).
- **Sampling**: this is a local, low-volume workshop pipeline (single Ollama
  process, no real traffic) — start at 100% sampling for all three
  evaluators; a real deployment with production volume would dial this down
  and rely on the offline dataset lane for full coverage instead.
- **Judge model (decided)**: **Amazon Bedrock** as the primary judge,
  provisioned as a Langfuse
  [LLM Connection](https://langfuse.com/docs/administration/llm-connection)
  (Bedrock is a natively supported provider; requires AWS credentials with
  `bedrock:InvokeModel` / `bedrock:InvokeModelWithResponseStream`). A Claude
  model on Bedrock is the working assumption — different family from the
  local `gemma4:e2b` generator, avoiding the ~25% self-enhancement bias
  inflation Loka pt. 2 found, and with reliable structured-output support,
  which the docs call essential for parsing scores. Exact model version is a
  cost/latency call to make when the connection is actually provisioned.

  **Local fallback (decided): a different Ollama model than the generator**,
  not `gemma4:e2b` itself — reusing the generator model as judge would quietly
  reintroduce the exact self-enhancement bias Bedrock was chosen to avoid,
  just during the periods we're already degraded (no internet). Register a
  second Langfuse LLM Connection pointed at the local Ollama server's
  OpenAI-compatible endpoint (`http://localhost:11434/v1`, via the "OpenAI"
  provider adapter with a custom Base URL — see docs' "Connecting via a
  gateway" section) using a model distinct from `OLLAMA_MODEL`.

  **How the fallback actually engages differs by lane** — Langfuse evaluators
  don't support automatic runtime failover between two LLM Connections:
  - *Online lane (this section)*: no built-in failover. If Bedrock calls
    start erroring/retrying (visible as `Delayed`/`Error` executions in the
    `langfuse-llm-as-a-judge` debug traces), the operational response is to
    manually repoint the three evaluators at the local LLM Connection via
    the Evaluators UI/API (same rubric prompts work with either connection),
    then switch back once connectivity recovers. This is a manual runbook,
    not automatic — flag it as an ops step, not "set and forget."
  - *Offline/experiment lane (§6.2's single-SDK-call alternative)*: since
    that path is already custom code, implement real automatic fallback —
    try the Bedrock call, catch connection/timeout errors, retry against the
    local model. Record which model actually produced each score (e.g. in
    `Score.comment`/metadata) so a run scored partly by the fallback model
    isn't silently blended with Bedrock-scored runs during agreement
    analysis (§7) — treat fallback-scored items as lower-confidence and
    candidates for re-scoring once Bedrock is reachable again.

### 6.4 Offline lane — Datasets & Experiments (regression testing)

- Build a Langfuse **Dataset** (e.g. `hpi-eval-transcripts`) of representative
  and edge-case transcripts (varied ages, chief complaints, and — per Loka
  pt. 2's diversity-bias finding — varied patient demographics, since these
  surface in the HPI's opening line). `input` = transcript text.
- `expected_output` (decided: **include it**): our rubric grades against the
  transcript, not a reference HPI, so it isn't required for scoring itself —
  but attach a clinician-authored gold HPI to every dataset item anyway, via
  Langfuse [Corrections](https://langfuse.com/docs/observability/features/corrections)
  or directly as `expected_output`. It doubles as a **judge sanity check**
  (the gold HPI should itself score ~4/4/4 when run back through the judge;
  if it doesn't, that's a rubric/prompt bug, not a generator bug) and as a
  stable reference point for comparing future prompt revisions. This adds
  reviewer authoring time per dataset item — budget for it alongside the
  Annotation Queue scoring pass in §7.
- Run the pipeline as the experiment **task**; attach the three evaluators
  (or the single combined SDK judge, §6.2) to score each `DatasetRun`
  automatically.
- Wire into [Experiments in CI/CD](https://langfuse.com/docs/evaluation/experiments/experiments-ci-cd)
  so any edit to `prompts/hpi_prompt.txt` reruns the dataset and reports
  score deltas before merge — this is the natural regression gate for a repo
  where prompts are already tracked as files.

## 7. Calibration & human-in-the-loop workflow

Following Hamel's critique-shadowing loop and Loka pt. 2's "LLM scores first,
human refines" pattern:

1. **Build a calibration set**: start with ~30 transcripts (vary chief
   complaint, demographics, complexity), growing until new review rounds stop
   surfacing new failure modes.
2. **Human scoring**: a clinical reviewer scores each transcript's HPI output
   on the *same* three ScoreConfigs via a Langfuse
   [Annotation Queue](https://langfuse.com/docs/evaluation/evaluation-methods/annotation-queues),
   writing a short rationale for each score (this rationale is what becomes
   a few-shot example in §5, not just a label).
3. **Run the LLM judge** on the same items; compare.
4. **Agreement metrics** (ordinal-aware, since 0-4 is ordinal not
   categorical):
   - Quadratic-weighted Cohen's κ or Spearman correlation per dimension.
   - Within-1-point agreement rate (adjacent scores treated as acceptable).
   - If collapsing to a ≥3 "pass" gate for monitoring dashboards: precision/
     recall of judge-pass vs. human-pass, watching for class imbalance
     (Hamel's warning that raw agreement misleads when most items pass).
   - **Decided: single reviewer for calibration**, at least for now — no
     second clinical reviewer is available to double-score a subset, so we
     have no measured inter-rater baseline of our own. Use the literature
     figures as the plausibility bar instead: Loka pt. 2 reports ~85%
     LLM-human agreement against ~81% human-human agreement; Hamel reports
     >90% agreement achieved after ~3 iterations on a comparable setup.
     These are *plausibility checks*, not hard gates — without our own
     baseline we can't say "matches human-human agreement," only "roughly in
     line with reported ranges elsewhere." If a second reviewer becomes
     available later, add the double-scoring pass — it would materially
     tighten this bar.
5. **Iterate**: when the judge disagrees with the human, read the human's
   rationale, decide whether the rubric anchor was ambiguous or the judge
   erred, and feed the corrected example back into §5's few-shot block.
   Repeat until agreement is acceptable and stable.
6. **Recalibrate** whenever the HPI prompt, the generation model, or the
   judge model changes.

## 8. Bias mitigations (from Loka pt. 1 & 2)

- **Self-enhancement bias**: judge model ≠ generator model/family (§6.3).
- **Verbosity bias**: rubric anchors are written around clinical content
  (chronology, coverage, register), not length — a longer HPI should not
  score higher on Accuracy/Completeness unless it adds clinically relevant,
  transcript-supported content. Optionally cross-check with a simple
  length/structure code-evaluator (§4 note) to catch a judge that starts
  rewarding padding.
- **Position bias**: not applicable — this is a **pointwise** design (one
  output, no second candidate to compare against), matching Loka pt. 1's
  recommendation for this evaluation shape.
- **Diversity/demographic bias**: Loka pt. 2 found measurable disparities in
  LLM clinical recommendations by race/sex. The calibration set (§7) should
  deliberately span patient demographics, and rationale text should be
  periodically spot-checked for systematic differences in how the judge
  treats otherwise-equivalent cases.

## 9. Rollout plan

1. ~~Resolve the HPI-scope open question (§3)~~ — done, meds/allergies/
   social history are in-scope; rubric anchors locked.
2. Build the calibration set and run it through an Annotation Queue (§7).
3. Draft the three judge prompts (§5); provision **two** Langfuse LLM
   Connections — Bedrock (primary) and a second, non-generator local Ollama
   model (fallback) — per §6.3.
4. Measure agreement; iterate via critique shadowing until stable.
5. Confirm the live trace shape for the `hpi` observation (§6.3); wire the
   three observation-level evaluators at 100% sampling for local/dev
   monitoring.
6. Build the `hpi-eval-transcripts` dataset; wire evaluators to Experiments;
   add to CI so prompt edits get scored automatically.
7. Recalibrate after any prompt/model change; watch Score Analytics for
   drift over time.

## 10. Open questions / risks

- **Deferred, not yet resolved**: exact observation to target for the `hpi`
  node's input/output mapping (§6.3) — plan is to run the pipeline once with
  tracing on and inspect the resulting trace in the local Langfuse instance
  (`localhost:3000`, project `pycon`) to confirm whether the node span or the
  nested LLM generation carries `input = transcript`. Not yet done.
- **Still open**: no AWS credentials for Bedrock are configured in this repo
  yet — provisioning them (plus the Bedrock Langfuse LLM Connection, and a
  second local Ollama model pulled for the fallback connection) is a
  prerequisite before any evaluator can actually run (§6.3).
- **Still open**: exact Bedrock model version (cost/latency call) and exact
  fallback Ollama model name are unpicked — placeholders in §6.3 until those
  connections are actually provisioned.
- **Still open**: the online lane's Bedrock→local fallback is a manual
  runbook, not automatic (§6.3) — no monitoring/alert is defined yet for
  *when* to trigger the manual switch (e.g. N consecutive `Error`/`Delayed`
  evaluator executions). Worth defining before this goes beyond local/dev use.
- **Accepted limitation**: calibration proceeds with a single reviewer (§7)
  — there is no measured inter-rater baseline of our own; agreement targets
  lean on literature figures instead. Revisit if a second reviewer becomes
  available.
- **Known issue, deferred by request**: `data/golden/golden_encounter_1.json`
  and `golden_encounter_2.json` store `physical_exam` as a single free-text
  string, not a per-body-system breakdown. `evals/eval_clinical_note_dataset.py`'s
  `pe_precision_recall_evaluator` needs the latter, so it still reads from a
  hand-authored `PE_SYSTEM_REFERENCE` dict in that script instead of the
  golden JSON files. Fix: restructure `physical_exam` in both golden files
  into a list of `{"system", "findings"}` entries (matching
  `models.PhysicalExamFinding`), then delete `PE_SYSTEM_REFERENCE` and read
  `golden["physical_exam"]` directly.
- **Known issue, deferred by request**: `evals/eval_trajectory.py` and
  `data/golden/golden_encounter_*.json`'s `expected_tool_calls` /
  `unexpected_tool_calls` reference tool names (`get_patient_history`,
  `lookup_icd10`) that don't match the real agent. The actual tools in
  `agent.py`/`icd10_search.py` are `search_icd10_codes` /
  `search_icd10_codes_batch`, and there is no `get_patient_history` tool at
  all — running `eval_trajectory` against a real trace will always report
  both as missing. Fix: either rename the real ICD-10 tool call sites (or add
  a name-mapping in `eval_trajectory.py`) to match `lookup_icd10`, and decide
  whether `get_patient_history` should be implemented as a real tool or
  stripped from the golden data's `expected_tool_calls`.


4. Convert both marimo notebooks to ipynb format and add them to the repo. Remove marimo dependencies.

Generate and modify the current readme with the following sections at least:
- Arquitectura: Incluyendo diagramas del agente y evaluación [AQUI PODEMOS AGREGAR EL DE MAFE DE EXCALIDRAW, añade un espacio]
- Prerequisitos: Todos los prerequisitors para correr el workshop:
  - Docker Desktop (or Docker Engine with Compose v2 — the docker compose command, with a space)
  - 15GB of free disk space for local LLM models and docker images
  - UV
  - Docker compose v2 (or Docker Desktop with Compose v2)
  - Acceso a Bedrock 
- Instrucciones de instalación: Cómo instalar y configurar el entorno para correr el workshop
  - docker compose up --build -d for langfuse
  - uv sync
  - descarga de los modelos con ollama
- Paso a paso del Workshop
  - Building an agent: Explicación de cómo construir un agente autónomo con el flujo de trabajo determinista
  - LLM as a judge: Explicación de cómo usar LLM como juez para evaluar HPI y examen físico, incluyendo la estructura de evaluación y cómo se integra con Langfuse
- Autores:
  - Nicolas Roldan - ML Engineer @ Loka
  - Mafe Castaño - ML Engineer @ Loka

All writen in spanish except for technical terms related to agents such as judge and the library names, including the metrics names.