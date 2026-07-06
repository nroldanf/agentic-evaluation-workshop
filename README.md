# Agents Evaluation Workshop

A minimal LangGraph medical-scribe agent. It reads a patient–doctor transcript,
uses a local Ollama model (`qwen3.5:4b` by default) to extract a structured
clinical note, and saves it to `outputs/`. Vitals and diagnoses are extracted by
separate nodes that can run sequentially (default) or in parallel.

## Prerequisites

1. Install [Ollama](https://ollama.com) and pull the model:

   ```bash
   ollama pull qwen3.5:4b
   ```

2. Install dependencies with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync
   ```

## Run the agent

Run against the sample transcript (defaults shown):

```bash
uv run python agent.py
```

This reads `data/transcript.txt`, prints the extracted `ClinicalNote`, and
writes it to `outputs/clinical_note.json`.

Pass your own transcript, a different user prompt, and/or an output path:

```bash
uv run python agent.py path/to/transcript.txt
uv run python agent.py path/to/transcript.txt -d prompts/diagnoses_prompt.txt
uv run python agent.py path/to/transcript.txt -o outputs/my_note.json
```

### ICD-10 code validation

Add `--use-tool` to give the agent an in-memory ICD-10-CM fuzzy-search tool so it
can look up and validate diagnosis codes. This also switches the default
diagnoses prompt to `prompts/diagnoses_prompt_with_tool.txt`.

```bash
uv run python agent.py --use-tool
```

### Sequential vs. parallel execution

By default the vitals and diagnoses nodes run **sequentially** (one Ollama call
at a time), which is reliable against a single local server. Add `--parallel` to
run them concurrently — this requires a local Ollama server configured for
concurrency (`OLLAMA_NUM_PARALLEL >= 2`), otherwise the simultaneous calls can
return empty responses.

```bash
uv run python agent.py --parallel
```

See all options with `uv run python agent.py --help`.

## Configuration

The Ollama model is configurable via the `OLLAMA_MODEL` environment variable
(defaults to `qwen3.5:4b`). Set it in `.env` (see `.env.example`) or inline:

```bash
OLLAMA_MODEL="llama3.1:8b" uv run python agent.py
```

## Prompts

All prompts are read from files under `prompts/`, so you can edit them without
touching the code:

- `prompts/system_prompt.txt` — the agent's system prompt (role/instructions).
- `prompts/diagnoses_prompt.txt` — the diagnoses extraction instructions, combined
  with the transcript at run time. Override it per run with `-d/--diagnoses-prompt`.
- `prompts/diagnoses_prompt_with_tool.txt` — the diagnoses extraction instructions
  used when `--use-tool` is set (guides the agent through the ICD-10 tool).
- `prompts/vitals_prompt.txt` — the instructions for the dedicated vitals node.

## Tracing (optional)

Tracing with [Langfuse](https://langfuse.com) is enabled automatically when the
keys are set. Copy `.env.example` to `.env` and fill in your `LANGFUSE_*` values
(public/secret key and `LANGFUSE_HOST`). Without them, the agent runs normally
with tracing disabled.

## References to read later
- Constraint tax: https://arxiv.org/pdf/2606.25605 
