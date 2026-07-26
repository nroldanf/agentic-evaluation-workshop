# Agents Evaluation Workshop

Un agent minimalista de medical-scribe construido con LangGraph. Lee la
transcripción (transcript) de una consulta médico–paciente, usa un modelo local
de Ollama ([`qwen3.5:9b`](https://ollama.com/library/qwen3.5:9b) por defecto) para extraer una nota clínica estructurada
y la guarda en `outputs/`. La historia de la enfermedad actual (HPI), los signos
vitales (vitals), el examen físico (physical exam) y los diagnósticos se extraen
mediante nodes (nodos) separados, ejecutados de forma secuencial.

## Arquitectura

Cuatro nodes extraen la nota, cada uno con su propio prompt y schema de salida,
todos construidos con una única función `build_agent(prompt, response_format,
tools)`:

- **hpi node** (`HistoryOfPresentIllness`) — redacta la historia de la enfermedad
  actual como un párrafo narrativo cronológico. Sin tools.
- **vitals node** (`VitalSigns`) — extrae los signos vitales. Sin tools.
- **physical exam node** (`PhysicalExam`) — documenta los hallazgos **objetivos**
  del examen físico (lo observado/medido por el clínico, no los síntomas referidos
  por el paciente), agrupados por sistema corporal. El sistema se restringe a un
  enum fijo (`PhysicalExamSystem`: general, heent, neck, cardiovascular,
  respiratory, …). Sin tools.
- **diagnoses node** (`DiagnosesOutput`) — extrae los diagnósticos diferenciales
  (los 3 más relevantes) y el assessment. Con `--use-tool` valida los códigos
  contra el tool de búsqueda de ICD-10, mediante un agent con tool-calling (por
  defecto) o, con `--deterministic`, mediante un pipeline de tres pasos con
  búsqueda determinística (ver más abajo).

Sus salidas se combinan en un único `ClinicalNote`.

```mermaid
flowchart LR
    S([START]) --> H["hpi node<br/>(HistoryOfPresentIllness)"]
    H --> V["vitals node<br/>(VitalSigns)"]
    V --> P["physical exam node<br/>(PhysicalExam)"]
    P --> D["diagnoses node<br/>(DiagnosesOutput)"]
    D --> E([END])
    D -.->|"--use-tool"| T["search_icd10_codes_batch"]
    T -.-> D
```

### Diagnósticos determinísticos (`--deterministic`)

Con `--use-tool --deterministic`, el diagnoses node reemplaza el agent con
tool-calling por un pipeline fijo de tres pasos, para que la búsqueda de
ICD-10 ocurra exactamente una vez, en código, en vez de quedar a discreción
del modelo (que en la versión con tool-calling podía repetir o reformular la
búsqueda varias veces antes de converger):

```mermaid
flowchart LR
    S([START]) --> EC["extract_candidates<br/>(LLM, sin tools)"]
    EC --> SC["search_candidates<br/>(Python puro, una sola búsqueda)"]
    SC --> SD["select_diagnoses<br/>(LLM, con reasoning)"]
    SD --> E([END])
```

1. **`extract_candidates`** — un LLM sin tools lista cada diagnóstico candidato
   discutido o implícito en el transcript (tanto diferenciales como
   assessment), junto con un `search_term` corto para buscarlo.
2. **`search_candidates`** — código puro: busca cada `search_term` una única
   vez contra el catálogo de ICD-10 (sin LLM de por medio).
3. **`select_diagnoses`** — un LLM (con reasoning habilitado) recibe el
   transcript original junto a los resultados ya buscados de cada candidato, y
   elige el código correcto de entre esos resultados — o descarta el
   candidato si ninguno es un match genuino — y consolida el resultado final.

```bash
uv run python agent.py --use-tool --deterministic
```

## Prerequisites

Necesitas dos herramientas instaladas localmente:

- **[Ollama](https://ollama.com)** — para ejecutar el modelo local.
- **[uv](https://docs.astral.sh/uv/)** — para gestionar el entorno y las dependencias de Python.

1. Instala [Ollama](https://ollama.com) y descarga el modelo
   [`qwen3.5:9b`](https://ollama.com/library/qwen3.5:9b):

   ```bash
   ollama pull qwen3.5:9b
   ```

2. Instala las dependencias con [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync
   ```

## Ejecutar el agente

Ejecuta con el transcript de ejemplo (se muestran los valores por defecto):

```bash
uv run python agent.py
```

Esto lee `data/transcript.txt`, imprime el `ClinicalNote` extraído y lo escribe
en `outputs/clinical_note.json`.

Puedes pasar tu propio transcript, un diagnoses prompt distinto y/o una ruta de
salida:

```bash
uv run python agent.py path/to/transcript.txt
uv run python agent.py path/to/transcript.txt -d prompts/diagnoses_prompt.txt
uv run python agent.py path/to/transcript.txt -o outputs/my_note.json
```

### Validación de códigos ICD-10

Agrega `--use-tool` para darle al agent un tool de fuzzy-search de ICD-10-CM
(en memoria) para que pueda buscar y validar los códigos de diagnóstico. Esto
también cambia el diagnoses prompt por defecto a
`prompts/diagnoses_prompt_with_tool.txt`.

```bash
uv run python agent.py --use-tool
```

### Ejecutar solo algunos nodes

Los cuatro nodes son independientes entre sí (ninguno depende de la salida de
otro), así que podés correr solo un subconjunto con `--only`, o excluir
algunos con `--skip` — útil para iterar rápido sobre un node (p. ej. mientras
ajustás un prompt) sin pagar el costo de correr los demás:

```bash
uv run python agent.py --only hpi
uv run python agent.py --skip diagnoses
```

Los nodes que no corren quedan con su valor por defecto/vacío en el
`ClinicalNote` resultante (p. ej. `hpi=""`, `assessment` vacío).

### Caching de nodos

Agrega `--cache` para cachear en disco el resultado de cada node. En una
ejecución posterior con las mismas entradas (mismo transcript, mismo prompt y
mismo modelo), ese node **omite la llamada al LLM** y devuelve el resultado
guardado — útil para re-correr el pipeline (p. ej. durante evaluaciones) sin
pagar de nuevo las llamadas lentas al modelo local.

```bash
uv run python agent.py --cache
```

La clave de caché de cada node incluye el transcript, el prompt efectivo de ese
node (rol del sistema + prompt de la tarea), el modelo y la temperatura. Por eso,
**editar un prompt re-ejecuta solo el node afectado**; los demás siguen sirviendo
desde caché. El cache es persistente entre procesos (un `DiskCache` que implementa
la interfaz `BaseCache` de LangGraph, en `cache.py`), a diferencia del
`InMemoryCache` integrado, que no sobrevive al proceso.

Para borrar todos los resultados cacheados:

```bash
uv run python agent.py --clear-cache
```

Consulta todas las opciones con `uv run python agent.py --help`.

## Configuration

El comportamiento del agent se configura mediante variables de entorno.
Defínelas en `.env` (ver `.env.example`) o de forma inline:

- `OLLAMA_MODEL` — el modelo de Ollama a usar (por defecto
  [`qwen3.5:9b`](https://ollama.com/library/qwen3.5:9b)).
- `OLLAMA_TEMPERATURE` — temperatura de sampling (por defecto `0.1`; más baja =
  más determinista).
- `OLLAMA_TIMEOUT` — timeout por request al servidor de Ollama, en segundos (por
  defecto `300`). Súbelo si un modelo local lento supera el tiempo límite.
- `LLM_MAX_RETRIES` — reintentos de una llamada al modelo fallida, p. ej. una
  respuesta estructurada vacía, antes de rendirse (por defecto `5`; usa backoff
  exponencial).
- `LANGGRAPH_CACHE_DIR` — directorio del cache de nodos en disco, usado con
  `--cache` (por defecto `.cache/scribe`).
- `LANGGRAPH_CACHE_TTL` — TTL en segundos para los resultados cacheados
  (opcional; vacío = sin expiración, un hit se sirve solo con entradas idénticas).

```bash
OLLAMA_MODEL="llama3.1:8b" uv run python agent.py
```

## Prompts

Todos los prompts se leen desde archivos en `prompts/`, así que puedes editarlos
sin tocar el código:

| Archivo | Usado por | Instrucciones |
| --- | --- | --- |
| `prompts/system_prompt.txt` | todos los nodes | Rol/instrucciones compartidos del scribe; se antepone al prompt de cada node para formar su system prompt. |
| `prompts/hpi_prompt.txt` | hpi node | Redacción de la historia de la enfermedad actual como párrafo narrativo. |
| `prompts/vitals_prompt.txt` | vitals node | Extracción de los signos vitales. |
| `prompts/physical_exam_prompt.txt` | physical exam node | Hallazgos del examen físico por sistema corporal. |
| `prompts/diagnoses_prompt.txt` | diagnoses node (sin `--use-tool`) | Extracción de diagnósticos (los 3 diferenciales más relevantes + assessment). Puedes sobrescribirlo por ejecución con `-d/--diagnoses-prompt`. |
| `prompts/diagnoses_prompt_with_tool.txt` | diagnoses node (`--use-tool`, sin `--deterministic`) | Guía al agent con tool-calling a través del tool de ICD-10. |
| `prompts/diagnoses_candidates_prompt.txt` | `extract_candidates` (`--use-tool --deterministic`) | Lista los diagnósticos candidatos y su `search_term`, sin asignar códigos. |
| `prompts/diagnoses_selection_prompt.txt` | `select_diagnoses` (`--use-tool --deterministic`) | Elige el código de cada candidato a partir de los resultados ya buscados, y consolida el resultado final. |

## Tracing (opcional)

El tracing con [Langfuse](https://langfuse.com) se habilita automáticamente
cuando las keys están definidas. Sin ellas, el agent se ejecuta normalmente
con el tracing deshabilitado.

### Levantar Langfuse localmente

El repo incluye un `docker-compose.yml` (basado en el
[self-hosting oficial de Langfuse](https://langfuse.com/self-hosting), v3) para
correr una instancia local completa: `langfuse-web`, `langfuse-worker`,
`postgres`, `clickhouse`, `redis` y `minio`. `.env.example` ya trae **todos**
los valores que necesita — infra secrets, keys y el seed de org/project/user —
así que no hay nada que generar ni configurar:

```bash
cp .env.example .env   # si aún no lo hiciste
docker compose up -d
```

Eso es todo. La UI queda en http://localhost:3000 — inicia sesión con
`user@example.com` / `langfuse`. El proyecto sembrado usa directamente tus
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` de `.env` como las keys del
proyecto (ver `docker-compose.yml`), así que el agent ya traza contra esta
instancia sin ningún paso extra.

Para bajar el stack:

```bash
docker compose down          # conserva los datos (volumes)
docker compose down -v       # borra también los volumes (reset completo)
```

Si preferís no auto-provisionar nada y crear el org/project/user a mano desde
la UI, comentá `LANGFUSE_INIT_ORG_ID` en `.env` — es el interruptor maestro:
sin él, Langfuse no crea nada aunque el resto de `LANGFUSE_INIT_*` esté
definido.

#### Generar tus propias keys y secrets (opcional)

Los defaults de `.env.example` son solo para una instancia local de un solo
uso. Si vas a compartir esta instancia con alguien más, o querés un proyecto
Langfuse propio en vez del dummy sembrado, genera tus propios valores.
Ningún valor real debe vivir en `docker-compose.yml` (queda trackeado en
git); todos van en tu `.env` local (gitignored). Comandos para generar cada
uno:

- **`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`** — el formato que valida
  Langfuse es `pk-lf-<uuid>` / `sk-lf-<uuid>`. Para arrancar un proyecto nuevo
  desde cero:

  ```bash
  echo "LANGFUSE_PUBLIC_KEY=pk-lf-$(uuidgen | tr 'A-Z' 'a-z')"
  echo "LANGFUSE_SECRET_KEY=sk-lf-$(uuidgen | tr 'A-Z' 'a-z')"
  ```

  Alternativa: dejar `LANGFUSE_INIT_*` sin definir, crear el proyecto a mano en
  la UI (http://localhost:3000) y copiar las keys que Langfuse genera ahí a tu
  `.env`.
- **`ENCRYPTION_KEY`** — 256 bits en hex: `openssl rand -hex 32`
- **`SALT`** y **`NEXTAUTH_SECRET`** — 256 bits en base64:
  `openssl rand -base64 32`
- **`POSTGRES_PASSWORD`**, **`CLICKHOUSE_PASSWORD`**, **`REDIS_AUTH`**,
  **`MINIO_ROOT_PASSWORD`**, **`LANGFUSE_INIT_USER_PASSWORD`** — cualquier
  password fuerte, p. ej.: `openssl rand -base64 18`

Después de generarlos, agrégalos a tu `.env` (no a `docker-compose.yml` ni a
`.env.example`) y reinicia el stack (`docker compose up -d`) para que tomen
efecto.

## Evals

Todos los scripts de evaluación viven en `evals/` (se corren siempre desde la
raíz del repo, no desde adentro de `evals/`, para que sus paths relativos a
`prompts/`/`data/` resuelvan bien):

- `evals/eval_hpi_judge.py` / `evals/eval_clinical_note_dataset.py` — LLM-as-a-Judge,
  requieren Langfuse + un juez (Bedrock u Ollama). Comparten la selección de
  modelo juez vía `evals/judge_client.py` (no se corre directamente).
- `evals/eval_diagnoses.py` / `evals/eval_vitals.py` / `evals/eval_trajectory.py` —
  funciones puras, sin Langfuse ni LLM.

### HPI LLM-as-a-Judge

`evals/eval_hpi_judge.py` califica un HPI ya generado (una traza existente en
Langfuse) contra su transcript, en tres dimensiones — 0 a 4 cada una, ver
`prompts/hpi_judge_prompt.txt` — **Accuracy** (fidelidad al transcript),
**Completeness** (cobertura del contenido relevante) y **Tone** (registro de
documentación clínica). Adjunta los resultados de vuelta a Langfuse como
Scores (`hpi_accuracy`, `hpi_completeness`, `hpi_tone`).

Requiere una traza existente con un node `hpi` — corré `agent.py` con
tracing habilitado (ver arriba) al menos una vez antes.

```bash
uv run python evals/eval_hpi_judge.py                        # usa la traza más reciente
uv run python evals/eval_hpi_judge.py --trace-id <trace_id>  # traza específica
```

#### Modelo juez

El juez usa un modelo distinto al `OLLAMA_MODEL` del generador a propósito —
evita el sesgo de que un modelo se auto-prefiera al calificarse a sí mismo:

- **Amazon Bedrock** (por defecto) — requiere credenciales AWS en el entorno
  (p. ej. `aws sso login`) con permiso `bedrock:InvokeModel` /
  `bedrock:InvokeModelWithResponseStream`.
- **Fallback a Ollama local** — si la llamada a Bedrock falla (sin internet,
  credenciales vencidas, sin acceso al modelo), reintenta automáticamente
  contra un modelo local distinto al del generador, y lo deja registrado en
  el `comment`/metadata del score.

Variables de entorno (ver `.env.example`, sección `HPI LLM-AS-A-JUDGE`):

- `JUDGE_BEDROCK_MODEL_ID` — model id o inference profile de Bedrock (por
  defecto `us.anthropic.claude-sonnet-4-5-20250929-v1:0`).
- `AWS_REGION` — región de Bedrock (por defecto `us-east-1`).
- `JUDGE_FALLBACK_OLLAMA_MODEL` — modelo local de respaldo (por defecto
  `mistral:latest`); debe ser distinto de `OLLAMA_MODEL`.

### Clinical Note Dataset Eval (HPI + Physical Exam)

`evals/eval_clinical_note_dataset.py` es el carril offline/regresión: corre la
extracción de HPI + Physical Exam para cada encuentro golden y adjunta 8
evaluators (LLM-judge + code) a un mismo Experiment/DatasetRun en Langfuse
(ver el docstring del archivo para el detalle de qué mide cada score).

Los encuentros golden se leen de `data/golden/golden_encounter_*.json`
(emparejados con su transcript en `data/encounter_*.txt` por `encounter_id`,
p. ej. `"encounter_id": "RIV-001"` -> `data/encounter_riv001.txt`) — se pasan
como argumento en vez de estar hardcodeados en el script:

```bash
uv run python evals/eval_clinical_note_dataset.py                                                   # los 2 encuentros default
uv run python evals/eval_clinical_note_dataset.py --golden-file data/golden/golden_encounter_riv001.json  # solo uno (repetible)
uv run python evals/eval_clinical_note_dataset.py --run-name mi-corrida-de-prueba
```

Requiere Langfuse corriendo (ver arriba) y el mismo juez que `evals/eval_hpi_judge.py`
(Bedrock por defecto, fallback a Ollama local).

> **Nota:** `pe_precision`/`pe_recall` (breakdown del Physical Exam por
> sistema corporal) todavía se comparan contra una referencia hardcodeada en
> `PE_SYSTEM_REFERENCE` dentro del script, en vez de leerla de
> `golden_encounter_*.json` — ese archivo guarda `physical_exam` como texto
> libre, no como lista por sistema. Ver TODO.md §10 ("Open questions / risks").

### Evals determinísticos (diagnósticos, vitals, trayectoria)

`evals/eval_diagnoses.py`, `evals/eval_vitals.py` y `evals/eval_trajectory.py`
son funciones puras — sin llamadas a Langfuse ni a ningún LLM — que comparan
una nota clínica extraída (p. ej. `outputs/encounter_2.json`, generado por
`agent.py`) o una traza de tool calls contra un `golden_encounter_*.json`.
`evals/run_evals.py` es un stub vacío, pensado como futuro punto de entrada
CLI para correr los tres a la vez.

Por ahora se exploran celda a celda en `notebooks/eval_walkthrough.ipynb`,
que carga `data/golden/golden_encounter_{1,2}.json` junto con sus mocks
`extracted_encounter_{1,2}_{good,bad}.json` / `trace_encounter_{1,2}_{good,bad}.json`
y corre cada eval contra la versión "good" y la "bad" para mostrar qué señal
produce cada uno:

```bash
uv run --with jupyter jupyter notebook notebooks/eval_walkthrough.ipynb
```

(o abrilo directamente con la extensión de Jupyter de VS Code / PyCharm — es
un `.ipynb` estándar, no requiere nada especial). No necesita API key ni
internet: todo lee JSON local y el `.parquet` de catálogo ICD-10.

## References to read later
- Constraint tax: https://arxiv.org/pdf/2606.25605 
