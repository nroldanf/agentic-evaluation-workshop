# Ground-Truth Documentation URLs

Use these URLs as the sole authoritative sources for Langfuse. Always fetch the latest contents when working on related tasks.

**Fetch note:** In this environment `WebFetch` is blocked for `langfuse.com` (network/enterprise policy — verified). Fall back to `curl -s <url>` via Bash, which returns raw markdown for `.md` pages. Append `.md` to any docs page URL for clean markdown; the index files already include `.md` on every link.

## Phase 1 — Index files

The root `llms.txt` is a category map (page names only) plus MCP/skill pointers. The three sub-indexes below carry the actual `[Title](url.md): description` links — pick the one that matches the task domain.

| Resource | URL | When to Use |
|----------|-----|-------------|
| Root index (routing) | https://langfuse.com/llms.txt | Orientation: see all doc categories; find MCP-server & official-skill pointers. Names only, no per-page links. |
| Product docs index | https://langfuse.com/llms-docs.txt | Phase 1 for tracing/observability, evaluation, scores, LLM-as-a-Judge, datasets, experiments, prompt management, metrics, API/data platform, SDKs. **The common case.** |
| Integrations index | https://langfuse.com/llms-integrations.txt | Phase 1 for wiring Langfuse into frameworks, LLM providers, gateways, or tools (LangChain, LangGraph, Strands, OpenAI, Anthropic, Bedrock, LiteLLM, OpenTelemetry, n8n, etc.). |
| Self-hosting index | https://langfuse.com/llms-self-hosting.txt | Phase 1 for deploying/operating self-hosted Langfuse (Docker, Kubernetes/Helm, Terraform, ClickHouse, Postgres, S3, scaling, upgrades). |

## Phase 2 — Individual pages

| Resource | URL | When to Use |
|----------|-----|-------------|
| Docs page (raw markdown) | `https://langfuse.com/docs/{path}.md` | Phase 2: fetch specific pages. **Append `.md`** for clean raw markdown (index links already include it). |
| Docs page (rendered) | `https://langfuse.com/docs/{path}` | Same content rendered; use if the `.md` form ever fails. |

## SDK / API reference (verify signatures — don't guess)

Use these to confirm exact class/function names, constructor params, and import paths before writing code.

| Resource | URL | When to Use |
|----------|-----|-------------|
| Python SDK reference | https://python.reference.langfuse.com/ | Verify Python SDK signatures (note: Python v2 vs v3 differ significantly). |
| JS/TS SDK reference | https://js.reference.langfuse.com/ | Verify JS/TS SDK signatures (v3/v4/v5). |
| Public HTTP API | Fetch `https://langfuse.com/docs/api-and-data-platform/features/public-api.md` (Phase 2) | REST endpoints for ingestion and querying. |

## Docs MCP Server (official grounding alternative)

Langfuse publishes an MCP server that exposes tools to search docs, GitHub issues, and discussions.

| Field | Value |
|-------|-------|
| Endpoint | `https://langfuse.com/api/mcp` |
| Transport | `streamableHttp` |
| Setup docs | https://langfuse.com/docs/docs-mcp.md |

If this MCP server is connected to the session, it is a legitimate substitute for manual `.md` fetching. Otherwise, use the two-phase fetch strategy above.

## Notes

- **No `llms-full.txt`** for Langfuse (404, verified). There is no single full-docs dump — always use index + per-page `.md` fetches.
- Langfuse also ships an official first-party skill at https://github.com/langfuse/skills/tree/main/skills/langfuse — a useful cross-reference, but this Loka skill is the operative workflow here.
