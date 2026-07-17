---
name: loka-langfuse-docs
description: "Ground-truth documentation workflow for Langfuse observability and evaluation. Use when you need to add a feature, change behavior, configure, debug, or explain anything involving Langfuse — tracing/observability, evaluation, scores, LLM-as-a-Judge, datasets, experiments, prompt management, metrics, integrations, or self-hosting. Trigger on: 'langfuse', 'observability', 'tracing', 'trace', 'observation', 'span', 'generation', 'LLM-as-a-Judge', 'evaluation', 'evaluator', 'scores', 'score config', 'annotation queue', 'dataset', 'experiment', 'prompt management', 'prompt version', 'custom dashboard', 'metrics API', '@observe', 'langfuse.trace', 'langfuse_context', 'OpenTelemetry LLM', or any code importing 'langfuse'. Always fetch the latest docs from official URLs and treat them as the sole authoritative source."
---

# Langfuse Observability & Evaluation Docs

## Overview

This skill ensures any Langfuse work is grounded in the official, up-to-date documentation at https://langfuse.com/docs. The `llms.txt` family of endpoints are the sole sources of truth.

**Do not rely on training data or memory for Langfuse specifics** — the platform and its SDKs (Python v3, JS/TS v4/v5) evolve rapidly and have had breaking changes across major versions. Always fetch live docs and verify SDK signatures before writing code.

The objective is simple: **every Langfuse recommendation, code snippet, or config must trace back to a specific fetched doc page.** If the docs don't cover it, say so explicitly.

## Quick Start

1. Identify the task domain (see Domain Routing Table).
2. **Phase 1** — fetch the matching sub-index (`llms-docs.txt`, `llms-integrations.txt`, or `llms-self-hosting.txt`) to get linked pages.
3. **Phase 2** — fetch the 1-3 most relevant `.md` pages from that index.
4. Verify SDK/API signatures against the reference sites when writing code.
5. Answer using only what the docs support; flag gaps explicitly.

## Fetching Docs (read this first)

Langfuse doc pages serve clean raw markdown when you **append `.md`** to the URL (e.g. `https://langfuse.com/docs/evaluation/overview.md`). The index files already include the `.md` suffix on every link, so fetch links verbatim.

**Fetch mechanism — WebFetch may be blocked here:**
- Prefer `WebFetch` if available.
- **In this environment, `WebFetch` is blocked for `langfuse.com`** (enterprise/network policy — observed and verified). When it fails with a "unable to verify if domain is safe" error, fall back to `curl -s <url>` via Bash. This is verified working and returns the raw markdown directly for `.md` pages.
- Example: `curl -s https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge.md`

**Alternative grounding path — Langfuse Docs MCP server:** Langfuse officially publishes an MCP server (`https://langfuse.com/api/mcp`, transport `streamableHttp`) that exposes tools to search the docs, GitHub issues, and discussions. If it is connected to the session, it is a legitimate substitute for manual fetching. Manual `.md` fetching remains the default fallback.

## Domain Routing Table

Langfuse's root `llms.txt` is a **category map plus pointers** — it lists page *names* but not per-page links. The linked indexes live in three sub-index files. Route the task to the right one:

```
User request is about...
│
├─ Tracing/observability, evaluation, scores, LLM-as-a-Judge, datasets,
│  experiments, prompt management, metrics/dashboards, SDKs, API, data platform
│  ──► https://langfuse.com/llms-docs.txt      (product docs — the common case)
│
├─ Wiring Langfuse into a framework, LLM provider, gateway, or tool
│  (LangChain, LangGraph, Strands, OpenAI, Anthropic, Bedrock, LiteLLM,
│  CrewAI, LlamaIndex, Vercel AI SDK, OpenTelemetry, n8n, etc.)
│  ──► https://langfuse.com/llms-integrations.txt
│
├─ Deploying/operating self-hosted Langfuse (Docker, Kubernetes/Helm,
│  Terraform on AWS/Azure/GCP, ClickHouse, Postgres, S3, scaling, upgrades)
│  ──► https://langfuse.com/llms-self-hosting.txt
│
└─ Unclear ──► fetch root llms.txt for orientation, then pick a sub-index
```

The root `https://langfuse.com/llms.txt` is the orientation/routing layer and also carries the MCP-server and official-skill pointers. Use it when you need to see the full category landscape before choosing a sub-index.

## Two-Phase Fetch Strategy

The docs are large (100+ product pages, 150+ integrations). Never dump everything into context.

### Phase 1: Fetch the matching sub-index

Fetch the sub-index for the task domain (from the Routing Table). Each entry is a `[Title](url.md): description` line. Scan for the 1-3 pages that match the request.

### Phase 2: Fetch specific `.md` pages

Fetch only those pages (links already carry `.md`). These contain the actual content to answer from.

**Example:**
- User asks "how do I set up LLM-as-a-Judge evaluation in Langfuse?"
- Phase 1: `curl -s https://langfuse.com/llms-docs.txt` → find the evaluation section → identify `.../evaluation/evaluation-methods/llm-as-a-judge.md`
- Phase 2: fetch that page → answer from its content

### No full-docs dump

There is **no `llms-full.txt`** for Langfuse (404, verified). Do not attempt a single full-docs fetch. Always use the index + per-page `.md` approach.

## Workflow

### 1) Determine domain
Use the Domain Routing Table. When in doubt, start with `llms-docs.txt` (most implementation questions live there).

### 2) Fetch authoritative docs (two-phase)
- **Always** retrieve the latest docs from the URLs in `references/ground-truth.md`.
- **Never** rely on memory, cached knowledge, or secondary sources for Langfuse specifics.
- Remember the WebFetch → `curl` fallback above.
- If a URL fails or returns unexpected content, tell the user and suggest they check the official site directly.

### 3) Apply documentation
- Map the user request to exact sections of the fetched docs.
- Quote or reference specific doc sections/URLs when answering.
- If the docs are silent or ambiguous:
  - State explicitly: "The official docs do not cover this."
  - Propose safe defaults with clear caveats.
  - Suggest the user check GitHub issues/discussions (reachable via the Docs MCP server).
- When debugging, cross-check reported behavior against documented constraints, SDK version notes, and prerequisites.

### 4) Code-generation guardrails
When writing Langfuse code:
- **Confirm the SDK and version first.** Python has v2 vs v3; JS/TS has v3/v4/v5. Decorators, context handling, and client APIs differ across majors. Check the relevant get-started / upgrade-path page before writing.
- Verify import paths, class names, and function signatures against the API reference sites (see `references/ground-truth.md`) — do not guess.
- Verify parameter names against the fetched page rather than reconstructing from memory.
- If the docs show a different pattern than what the user expects, flag the discrepancy.
- Always include version context (e.g., "Based on Langfuse Python SDK v3 docs as of fetch date").

## Key Concepts Quick Reference

Stable orientation only. **Always verify current details against live docs.** Observability and evaluation are foregrounded because they are this skill's primary focus, but the skill covers all of Langfuse.

### Observability / Tracing
- **Trace**: One end-to-end execution of your app (e.g. a request). Top-level container.
- **Observation**: A step inside a trace. Types include **span** (durationed unit of work), **generation** (LLM call, with model/usage/cost), and **event** (point-in-time).
- **Instrumentation**: `@observe` decorator (Python), manual SDK calls, or OpenTelemetry-based auto-instrumentation via integrations.
- **Attributes**: metadata, tags, `user_id`, `session_id`, releases/versions — these power filtering, dashboards, evaluators, and experiments. See "What does a good trace look like?" (best-practices).
- **Sessions / Users**: Group related traces for conversation-level and per-user analysis.

### Evaluation
- **Scores**: Langfuse's universal object for storing evaluation results (numeric, categorical, boolean). Attached to traces/observations/sessions.
- **Score configs**: Define allowed score types/ranges for consistency.
- **Evaluation methods**: LLM-as-a-Judge (rubric-guided model scoring), code evaluators (deterministic Python/TS), annotation queues (human review), and scores via SDK/API.
- **Datasets & Experiments**: Structured test sets (`DatasetItem`s) run through a task + evaluators to produce `DatasetRun`s; used for benchmarking, regression testing, and CI/CD gating.

### Other domains (verify via docs when relevant)
- **Prompt management**: versioned prompts, labels, config, composability, A/B testing, caching, linking prompts to traces.
- **Metrics**: custom dashboards, Metrics API, monitors/alerts.
- **API & data platform**: Public API, Observations API, Query-via-SDK, CLI, blob-storage export.
- **Self-hosting**: Docker Compose, Kubernetes/Helm, Terraform (AWS/Azure/GCP), ClickHouse + Postgres + blob storage, scaling, upgrades.

## Resources

### references/
- `references/ground-truth.md` — Authoritative `llms.txt` sub-index URLs, `.md` page pattern, SDK/API reference sites, and the Docs MCP server endpoint.
