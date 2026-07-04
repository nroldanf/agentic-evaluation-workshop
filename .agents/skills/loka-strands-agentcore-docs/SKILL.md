---
name: loka-strands-agentcore-docs
description: "Ground-truth documentation workflow for Strands-Agents and Amazon Bedrock AgentCore. Use when you need to add a feature, change behavior, configure, debug, or explain anything involving Strands-Agents or AgentCore — including their services, SDKs, APIs, tools, or deployment. Trigger on: 'strands', 'strands-agents', 'agent loop', 'tool-calling', 'MCP tools', 'multi-agent', 'agent orchestration', 'AgentCore', 'AgentCore Runtime', 'AgentCore Memory', 'AgentCore Gateway', 'bedrock agents', 'agent deployment', 'strands agent', or any code importing 'strands'. Always fetch the latest docs from official URLs and treat them as the sole authoritative source."
---

# Strands-Agents & AgentCore Docs

## Overview

This skill ensures any Strands-Agents or Bedrock AgentCore work is grounded in the official, up-to-date documentation. The `llms.txt` and `llms-full.txt` endpoints are the sole sources of truth for these frameworks.

**Do not rely on training data or memory for Strands/AgentCore specifics** — these frameworks evolve rapidly. Always fetch live docs.

## Quick Start

1. Identify whether the task is about **Strands-Agents**, **AgentCore**, or **both** (see Scope Decision Tree).
2. Load URLs from `references/ground-truth.md`.
3. Use the **two-phase fetch strategy** (see below) to get only what you need.
4. Answer using only what the docs support; flag gaps explicitly.

## Scope Decision Tree

```
User request mentions...
│
├─ Agent code, SDK, Python/TS, tool-calling, agent loop,
│  prompts, hooks, streaming, multi-agent patterns,
│  model providers, plugins, evals, conversation management
│  ──► Strands-Agents docs
│
├─ Deployment, runtime, gateway, memory service, policy engine,
│  identity, VPC, IAM, CloudFormation, Terraform,
│  infrastructure, scaling, async processing
│  ──► AgentCore docs
│
├─ "Deploy a Strands agent to AgentCore", integration,
│  migration, end-to-end architecture
│  ──► Both docs (Strands for agent code, AgentCore for infra)
│
└─ Unclear ──► Ask the user to clarify scope
```

## Two-Phase Fetch Strategy

The docs are large (500+ URLs for Strands, 90+ for AgentCore). Never dump everything into context.

### Phase 1: Fetch the index

Fetch the `llms.txt` URL for the relevant framework. This returns a structured index of all documentation pages with brief descriptions. Scan it to identify which specific sub-pages are relevant to the user's question.

### Phase 2: Fetch specific sub-pages

Fetch only the 1-3 most relevant sub-page URLs from the index. These contain the actual documentation content needed to answer the question.

**Example:**
- User asks about "how to create a custom tool in Strands"
- Phase 1: Fetch `https://strandsagents.com/latest/llms.txt` → find the tools section → identify `concepts/tools/custom-tools/` URL
- Phase 2: Fetch that specific page → answer from its content

### When to use `llms-full.txt`

Strands-Agents also publishes `llms-full.txt` which contains **all documentation inline** (~15k+ words). Use it when:
- The user's question spans multiple topics and you need broad context
- You need to search across the full docs for a specific term or concept
- Phase 1 + Phase 2 didn't surface the answer

**Do NOT fetch `llms-full.txt` as the default** — it's very large and wastes context on irrelevant sections.

## Workflow

### 1) Determine scope

Use the Scope Decision Tree above. When in doubt, start with Strands docs (most common case for implementation questions).

### 2) Fetch authoritative docs (two-phase)

- **Always** retrieve the latest docs from URLs in `references/ground-truth.md`.
- **Never** rely on memory, cached knowledge, or secondary sources for Strands/AgentCore specifics.
- If a URL fails or returns unexpected content, inform the user and suggest they check the official site directly.

### 3) Apply documentation

- Map the user request to exact sections of the fetched docs.
- Quote or reference specific doc sections when answering.
- If the docs are silent or ambiguous on the user's question:
  - State explicitly: "The official docs do not cover this."
  - Propose safe defaults with clear caveats.
  - Suggest the user check GitHub issues or community channels.
- When debugging, cross-check reported behavior against documented constraints, version notes, or prerequisites.

### 4) Code generation guardrails

When writing Strands or AgentCore code:
- Verify import paths against the API reference in the docs.
- Verify function signatures and parameter names — do not guess.
- If the docs show a different pattern than what the user expects, flag the discrepancy.
- Always include the framework version context (e.g., "Based on Strands SDK latest docs as of fetch date").

## Key Concepts Quick Reference

These are stable architectural concepts to help with initial orientation. **Always verify current details against live docs.**

### Strands-Agents Core Concepts
- **Agent loop**: The core execution cycle — model call → tool use → model call → ...
- **Tools**: Functions the agent can call. Types: custom (Python decorator), MCP (Model Context Protocol), vended (AWS-provided), community.
- **Model providers**: Bedrock (default), Anthropic, OpenAI, Google, Ollama, LiteLLM, and many more.
- **Plugins**: Extend agent behavior (skills, steering).
- **Multi-agent**: Patterns include agent-as-tool, swarm, graph, workflow.
- **Session management**: Persist conversation state across invocations.
- **Observability**: Built-in metrics, traces, and logs via OpenTelemetry-compatible interfaces.
- **Evals SDK**: Evaluate agent quality with trajectory, tool selection, faithfulness, and custom evaluators.

### AgentCore Core Concepts
- **Runtime**: Managed compute for running agents at scale.
- **Gateway**: API endpoint management and routing.
- **Memory**: Managed session and long-term memory service.
- **Policy Engine**: Governance and tool-use policies.
- **Identity**: Authentication and authorization for agents.
- **Built-in tools**: Browser tool, code interpreter.
- **Deployment modes**: Direct, async processing.
- **IaC support**: CloudFormation and Terraform templates.

## Resources

### references/
- `references/ground-truth.md` — Authoritative `llms.txt` and `llms-full.txt` URLs.
