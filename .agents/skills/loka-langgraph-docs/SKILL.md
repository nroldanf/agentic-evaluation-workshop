---
name: loka-langgraph-docs
description: "Ground-truth documentation workflow for building LangGraph agents. Use when you need to add a feature, change behavior, configure, debug, or explain anything involving LangGraph — its graph API, state, persistence, prebuilt agents, or deployment. Trigger on: 'langgraph', 'StateGraph', 'CompiledStateGraph', 'graph API', 'nodes and edges', 'reducers', 'checkpointer', 'persistence', 'interrupts', 'human-in-the-loop', 'create_react_agent', 'prebuilt agent', 'subgraphs', 'Command', 'Send', 'streaming', 'time travel', 'LangGraph Platform', or any code importing 'langgraph'. Always fetch the latest docs from official URLs and treat them as the sole authoritative source."
---

# LangGraph Docs

## Overview

This skill ensures any LangGraph work is grounded in the official, up-to-date documentation. The LangGraph `llms.txt` index and the per-page docs on `docs.langchain.com` are the sole sources of truth.

**Do not rely on training data or memory for LangGraph specifics** — the framework and its docs evolve rapidly (the docs recently moved to `docs.langchain.com`, and APIs like `create_react_agent`, `Command`, and the checkpointer interfaces change between versions). Always fetch live docs.

## Quick Start

1. Confirm the task is about **LangGraph** (the OSS Python/JS graph framework) vs. adjacent products (see Scope Decision Tree).
2. Load URLs from `references/ground-truth.md`.
3. Use the **two-phase fetch strategy** (see below) to get only what you need.
4. Answer using only what the docs support; flag gaps explicitly.

## Scope Decision Tree

```
User request mentions...
│
├─ Graph construction, StateGraph, nodes/edges, state schemas,
│  reducers, conditional routing, Command/Send, streaming,
│  persistence/checkpointers, memory, interrupts/human-in-the-loop,
│  subgraphs, time travel, prebuilt agents (create_react_agent),
│  tool-calling, multi-agent graphs
│  ──► LangGraph OSS docs (docs.langchain.com/oss/python/langgraph)
│
├─ Exact class/function signatures, constructor params,
│  return types, module paths
│  ──► LangGraph API reference (reference.langchain.com/python/langgraph)
│
├─ Deployment, LangGraph Server, LangGraph CLI, hosted platform,
│  assistants API, cron, LangSmith tracing/eval of a graph
│  ──► LangGraph Platform / LangSmith docs (docs.langchain.com)
│
└─ Unclear ──► Ask the user to clarify scope
```

## Two-Phase Fetch Strategy

The docs are large. Never dump everything into context.

### Phase 1: Fetch the index

Fetch the LangGraph `llms.txt` index (see `references/ground-truth.md`). It returns a structured, LangGraph-scoped list of documentation pages with brief descriptions and their canonical `docs.langchain.com/oss/python/langgraph/...` URLs. Scan it to identify the specific sub-pages relevant to the request.

### Phase 2: Fetch specific sub-pages as raw markdown

Fetch only the 1-3 most relevant pages. **Append `.md` to the page URL to get clean raw markdown** — this is the preferred fetch form for LangGraph docs:

```
https://docs.langchain.com/oss/python/langgraph/{topic}.md
```

**Example:**
- User asks "how do I add persistence / memory to my graph"
- Phase 1: Fetch the `llms.txt` index → find the persistence entry → note its path (`.../langgraph/persistence`)
- Phase 2: Fetch `https://docs.langchain.com/oss/python/langgraph/persistence.md` → answer from its content

### Verifying signatures

For exact class names, constructor parameters, and module import paths (e.g. `StateGraph`, `MemorySaver`, `create_react_agent`, `Command`, `interrupt`), consult the **API reference** at `reference.langchain.com/python/langgraph/` — conceptual pages sometimes lag the exact signature.

### On full-docs dumps

Unlike some frameworks, LangGraph does **not** publish a practical single full-docs file. The legacy `llms-full.txt` on the github.io host 404s, and `docs.langchain.com/llms-full.txt` is platform-wide and too large to fetch (>10 MB). **Do not attempt a full-docs dump.** Always use the index + per-page `.md` fetches instead.

## Workflow

### 1) Determine scope

Use the Scope Decision Tree above. When in doubt, start with the LangGraph OSS docs (the most common case for implementation questions).

### 2) Fetch authoritative docs (two-phase)

- **Always** retrieve the latest docs from URLs in `references/ground-truth.md`.
- **Never** rely on memory, cached knowledge, or secondary sources for LangGraph specifics.
- If a URL fails or returns unexpected content, inform the user and suggest they check the official site directly. Note the docs migrated to `docs.langchain.com`; older `langchain-ai.github.io/langgraph/...` deep links may redirect or 404.

### 3) Apply documentation

- Map the user request to exact sections of the fetched docs.
- Quote or reference specific doc sections when answering.
- If the docs are silent or ambiguous on the user's question:
  - State explicitly: "The official docs do not cover this."
  - Propose safe defaults with clear caveats.
  - Suggest the user check GitHub issues or community channels.
- When debugging, cross-check reported behavior against documented constraints, version notes, or prerequisites.

### 4) Code generation guardrails

When writing LangGraph code:
- Verify import paths against the API reference (`langgraph.graph`, `langgraph.prebuilt`, `langgraph.checkpoint.*`, `langgraph.types`, etc.) — do not guess.
- Verify function signatures and parameter names against the API reference.
- Prefer documented patterns: define a typed state (often a `TypedDict`) with reducers where needed, build with `StateGraph`, add nodes/edges, `.compile()` with a checkpointer for persistence.
- If the docs show a different pattern than what the user expects, flag the discrepancy.
- Always include version context (e.g. "Based on LangGraph docs at docs.langchain.com as of fetch date").

## Key Concepts Quick Reference

These are stable LangGraph concepts for initial orientation. **Always verify current details against live docs.**

- **StateGraph**: The core builder. Parameterized by a user-defined state schema; must be `.compile()`d (into a `CompiledStateGraph`) before invocation.
- **State & reducers**: A shared, typed state object passed between nodes. Reducers (e.g. `add_messages`) define how node outputs merge into state instead of overwriting.
- **Nodes**: Functions (or runnables) that take the current state and return a partial state update.
- **Edges**: Connections between nodes — normal edges, conditional edges (routing functions), and entry points. `START` / `END` are the terminal sentinels.
- **Command & Send**: `Command` lets a node both update state and control routing in one return; `Send` dispatches dynamic fan-out (map) to a node with per-item state.
- **Persistence / checkpointers**: Checkpointers (e.g. `MemorySaver`, SQLite/Postgres savers) persist state per thread, enabling memory, resumption, and time travel.
- **Interrupts / human-in-the-loop**: `interrupt()` pauses a graph for human input; resume by re-invoking with a `Command(resume=...)`.
- **Streaming**: Multiple stream modes (`values`, `updates`, `messages`, `debug`) for surfacing intermediate progress.
- **Prebuilt agents**: `create_react_agent` and related helpers in `langgraph.prebuilt` for tool-calling ReAct-style agents without hand-building the graph.
- **Subgraphs & multi-agent**: Graphs composed as nodes within other graphs; patterns for supervisor/swarm multi-agent orchestration.
- **Time travel**: Inspect and rewind to prior checkpoints to fork execution from a past state.

## Resources

### references/
- `references/ground-truth.md` — Authoritative `llms.txt` index, per-page `.md` fetch pattern, and API reference URLs.
