# Known limitations

Things the scribe agent (`agent.py`) still struggles with, found while testing the
ICD-10 search tool (`icd10_search.py`) and its use in
`prompts/diagnoses_prompt_with_tool.txt`.

## Tool-enabled extraction can silently drop valid differentials

With `--use-tool`, the diagnoses node (model: `qwen3.5:9b`, reasoning enabled, up to 25
tool results per candidate) can drop clearly-discussed conditions from
`differential_diagnoses` entirely, while still correctly resolving the assessment.

Example: a transcript where a patient discusses a history of atrial fibrillation
("afib"), a COPD flare, and worsening acid reflux/GERD. Running just the diagnoses node
(`agent.diagnoses_node`) with the tool enabled returned:

```json
{
  "differential_diagnoses": [],
  "assessment": {
    "icd10_code": "I48.91",
    "diagnosis_name": "Unspecified atrial fibrillation"
  }
}
```

COPD and GERD vanished — not miscoded, just absent — despite both being unambiguously
discussed in the transcript.

**Root cause narrowed down, not fully identified.** Initial hypothesis was that the
model was extracting candidate names as abbreviations/colloquialisms ("afib", "COPD",
"acid reflux") in Step 1, and that those don't share enough vocabulary with the
ICD-10-CM description text for the search tool to find a match (see the next section) —
so Step 2 would find nothing, and Step 3 would drop the candidate per the "if none of
the returned codes is a genuine match, omit it" rule.

That hypothesis does not hold up: re-running Step 1 in isolation (tools off, reasoning
off, `extraction_model`) against the same transcript, the model already writes fully
spelled-out clinical terms even with the *original* prompt wording:

```
["Atrial fibrillation", "Chronic obstructive pulmonary disease (COPD) exacerbation", "Gastroesophageal reflux disease"]
```

So the candidate names reaching the search tool were already fine. The dropping happens
somewhere in the tool-calling/reasoning loop itself (Step 2/3 execution) — plausibly the
small model losing track of multiple candidates while juggling tool results under
reasoning + tool-call load, or the model over-focusing on the primary complaint and
truncating differentials. This needs a traced re-run (capturing the model's tool calls
and reasoning) to pin down further — not yet done.

A prompt tweak was still made (Step 1 now explicitly bans abbreviations/acronyms, e.g.
"atrial fibrillation" not "afib") since it's a low-risk clarity improvement, but it
should **not** be expected to fix this particular issue.

## Search tool has a hard vocabulary-overlap ceiling

Benchmarked in `notebooks/diagnosis_analysis.py` across 25 hand-verified diagnosis
queries, comparing fuzzy (`rapidfuzz.WRatio`), full-text (SQLite FTS5 + BM25), and hybrid
(Reciprocal Rank Fusion of both):

| Metric  | Fuzzy | FTS   | Hybrid (RRF) |
|---------|-------|-------|--------------|
| hit@1   | 28.0% | 32.0% | 32.0%        |
| hit@5   | 52.0% | 56.0% | 60.0%        |
| hit@10  | 64.0% | 64.0% | 64.0%        |
| hit@25  | 72.0% | 68.0% | 72.0%        |

Hybrid (now wired into `agent.py` as `search_icd10_hybrid`/`search_icd10_hybrid_batch`,
`limit=25`) matches or beats both individual methods on every metric, but **none of the
three methods can find a code when the query shares no literal vocabulary with the
target ICD-10-CM description** — abbreviations (afib, copd) and colloquialisms far from
clinical wording (acid reflux vs. "Gastro-esophageal reflux disease") fail identically
across all three. This is a shared ceiling, not a fuzzy-vs-FTS-vs-hybrid difference —
fixing it would need synonym/abbreviation expansion (e.g. a small alias dictionary or
embedding-based semantic search), which hasn't been attempted.

A related idea — using the ICD-10-CM hierarchical code-family structure (shared
prefixes) as a reranking signal — was tested and **rejected**: boosting by family
popularity moves an entire code family up or down together, so it can never let one
sibling (e.g. an "unspecified" leaf) overtake another sibling already ranked above it.
It measurably hurt several previously-correct queries and was dropped in favor of the
plain RRF hybrid above. A different form of the idea — injecting a family's missing
general code as a candidate when its siblings dominate the pool but it's absent — was
proposed but not implemented or validated.

## Tool-free extraction nodes call the LLM 3x per node instead of once

`hpi_node`, `vitals_node`, and `physical_exam_node` are all built via
`build_agent(extraction_model, prompt, <bare Pydantic model>)` — no `tools`, one
structured result expected — so each should need exactly one LLM call. Confirmed by
direct test: a single `physical_exam_node` invocation against `data/transcript.txt`
produced **3** separate `POST /api/chat` calls to Ollama (not 1), correctly-shaped
output notwithstanding:

```
17:16:22 | POST /api/chat  (call 1, ~3s after start)
17:16:39 | POST /api/chat  (call 2, ~17s later)
17:16:39 | POST /api/chat  (call 3, same second as call 2)
```

No errors or `ModelRetryMiddleware` retry logs appear in between — this isn't
failure-triggered retries, it's inherent to the default structured-output strategy.
`build_agent`'s own docstring already flags the mechanism: passing a bare Pydantic
model (rather than `ProviderStrategy(schema=...)`) uses `create_agent`'s default
tool-based structured-output strategy, which exposes the schema as a hidden "extract"
tool; the model can emit free-form reasoning/prose across one or more calls before it
finally invokes that tool, instead of doing it in one shot. The diagnoses node already
avoids this (it uses `ProviderStrategy(schema=DiagnosesOutput)` explicitly, "preferred
for the diagnoses agent" per the docstring) — the same fix (switching `hpi_node`,
`vitals_node`, and `physical_exam_node` to `ProviderStrategy`) has not been tried for
the other three nodes. Given each local Ollama call already runs tens of seconds, this
roughly triples the latency of 3 of the 4 extraction nodes.

## Physical exam system-level precision/recall is noisy across overlapping enum categories

`evals/eval_clinical_note_dataset.py` scores `physical_exam_node`'s output against a golden
system-set for `data/encounter_1.txt`/`encounter_2.txt`. A real run against the
existing `outputs/encounter_2.json` extraction scored **precision 0.25 / recall
0.33** — not primarily because content was missing, but because the model filed the
hand-wound exam under `musculoskeletal` and the face/hand burns under `heent`, while
the golden set (and `encounter_1`'s own extraction) uses `extremities` and `skin` for
equivalent findings. `prompts/physical_exam_prompt.txt` gives disambiguation examples
for a few systems ("lungs -> respiratory") but none for limb wounds/burns, which can
reasonably map to `skin`, `extremities`, or `musculoskeletal` depending on framing.
Tightening that prompt guidance (or adding a small system-alias equivalence table to
the evaluator) would likely raise both metrics without the underlying extraction
actually improving — not yet attempted.

# Langfuse tracing (local Docker instance, project: pycon)
LANGFUSE_SECRET_KEY = "sk-lf-ab67a772-4cbe-4297-8315-cd01f4955f3e"
LANGFUSE_PUBLIC_KEY = "pk-lf-10d6b496-129a-48c6-bf84-7239b2ccf9b5"
LANGFUSE_BASE_URL = "http://localhost:3000"

# Langfuse local instance auto-provisioning (docker-compose.yml). The project's
# public/secret key are read from LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY above
# (see docker-compose.yml), so they don't need to be repeated here.
LANGFUSE_INIT_ORG_ID="my-org"
LANGFUSE_INIT_ORG_NAME="My Org"
LANGFUSE_INIT_PROJECT_ID="pycon2026"
LANGFUSE_INIT_PROJECT_NAME="pycon2026"
LANGFUSE_INIT_USER_EMAIL="nicolas@loka.com"
LANGFUSE_INIT_USER_NAME="nico"
LANGFUSE_INIT_USER_PASSWORD="QOU64ZPgDqAuvihYyZkNVZIp"
# Ollama model to use for extraction
# OLLAMA_MODEL="gemma4:e2b" # works (takes too long after a few calls)
# it is calling the model several times and not advancing to next step
# it calls 148 the model in vitals node

OLLAMA_MODEL="qwen3.5:9b" # works, but diagnoses extraction is too slow and takes too many calls
# it calls the model several times to review the returned results
# doesn't use the tool when reasoning it is disabled
# it uses the tool when reasoning is enabled

# OLLAMA_MODEL="ornith:9b" # works up to diagnosis, then fails with KeyError: 'structured_response'
# OLLAMA_MODEL="granite4.1:8b" # does not support reasoning
# Sampling temperature (optional; defaults to 0.1). Lower = more deterministic.
OLLAMA_TEMPERATURE="0.1"
# Per-request timeout in seconds for the Ollama server (optional; defaults to 300).
# Raise it if a slow local model times out on long generations.
OLLAMA_TIMEOUT="300"
# Retries for a failed model call (e.g. empty structured-output response) before
# giving up (optional; defaults to 5). Uses exponential backoff.
LLM_MAX_RETRIES="3"
# Node-level caching (enabled per run with --cache).
# Directory for the on-disk node cache (optional; defaults to .cache/scribe).
LANGGRAPH_CACHE_DIR=".cache/scribe"
# TTL in seconds for cached node results (optional; empty/unset = no expiry).
# Leave empty so a cache hit is served only for truly identical inputs.
LANGGRAPH_CACHE_TTL=""

# HPI LLM-as-a-Judge (evals/eval_hpi_judge.py). Separate from OLLAMA_MODEL above so
# the judge is never the same model as the generator (self-enhancement bias).
# claude-sonnet-5 isn't enabled for this AWS account on Bedrock yet; sonnet-4-5
# is, via the us.-prefixed cross-region inference profile.
JUDGE_BEDROCK_MODEL_ID="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
AWS_REGION="us-east-1"
JUDGE_FALLBACK_OLLAMA_MODEL="mistral:latest"
