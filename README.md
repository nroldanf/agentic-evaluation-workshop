# Agents Evaluation Workshop

A minimal LangGraph medical-scribe agent. It reads a patient–doctor transcript,
uses a local Ollama model (`qwen3.5:4b`) to extract a structured clinical note,
and saves it to `outputs/`.

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
uv run python agent.py path/to/transcript.txt -u prompts/user_prompt.txt
uv run python agent.py path/to/transcript.txt -o outputs/my_note.json
```

See all options with `uv run python agent.py --help`.

## Prompts

Both prompts are read from files under `prompts/`, so you can edit them without
touching the code:

- `prompts/system_prompt.txt` — the agent's system prompt (role/instructions).
- `prompts/user_prompt.txt` — the extraction instructions, combined with the
  transcript at run time. Override it per run with `-u/--user-prompt`.

## Tracing (optional)

Tracing with [Langfuse](https://langfuse.com) is enabled automatically when the
keys are set. Copy `.env.example` to `.env` and fill in your `LANGFUSE_*` values
(public/secret key and `LANGFUSE_HOST`). Without them, the agent runs normally
with tracing disabled.
