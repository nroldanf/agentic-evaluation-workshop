# Ground-Truth Documentation URLs

Use these URLs as the sole authoritative sources for LangGraph. Always fetch the latest contents when working on related tasks.

## LangGraph (OSS, Python)

| Resource | URL | When to Use |
|----------|-----|-------------|
| llms.txt (index) | https://langchain-ai.github.io/langgraph/llms.txt | Phase 1: scan for relevant sub-pages. LangGraph-scoped index; its links resolve to canonical `docs.langchain.com` pages. |
| Docs page (raw markdown) | `https://docs.langchain.com/oss/python/langgraph/{topic}.md` | Phase 2: fetch specific pages. **Append `.md`** for clean raw markdown (e.g. `graph-api.md`, `persistence.md`, `streaming.md`, `interrupts.md`). |
| Docs page (rendered) | `https://docs.langchain.com/oss/python/langgraph/{topic}` | Same content rendered; use if the `.md` form ever fails. |
| API reference | https://reference.langchain.com/python/langgraph/ | Verify exact class/function signatures, constructor params, and module import paths. |

### Why the index lives on the github.io host

The canonical docs have moved to `docs.langchain.com`, but that domain publishes **no LangGraph-scoped `llms.txt`** — its `/llms.txt` is platform/LangSmith-focused and has no LangGraph section, and `/oss/python/langgraph/llms.txt` and `/oss/llms.txt` both 404 (verified). The `langchain-ai.github.io/langgraph/llms.txt` index remains current and its links point at the canonical `docs.langchain.com` pages, so it is the intentional Phase 1 source. **Do not "fix" this to a `docs.langchain.com` index URL — none exists.**

### No full-docs dump

There is no practical single full-docs file for LangGraph:
- `https://langchain-ai.github.io/langgraph/llms-full.txt` → 404 (verified).
- `https://docs.langchain.com/llms-full.txt` exists but is platform-wide and too large to fetch (>10 MB; exceeds fetch limits).

Always use the index + per-page `.md` fetches. Do not attempt a full-docs dump.

## LangGraph Platform / Server / LangSmith

For deployment, LangGraph Server, the LangGraph CLI, hosted assistants, and tracing/eval of graphs, start from the main platform docs index and drill into the relevant section:

| Resource | URL | When to Use |
|----------|-----|-------------|
| Platform docs llms.txt | https://docs.langchain.com/llms.txt | Phase 1 for LangGraph Platform, LangGraph Server/CLI, and LangSmith topics. |
| Docs page (raw markdown) | `https://docs.langchain.com/{path}.md` | Phase 2: fetch specific platform/LangSmith pages as raw markdown. |
