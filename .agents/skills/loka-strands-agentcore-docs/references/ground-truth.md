# Ground-Truth Documentation URLs

Use these URLs as the sole authoritative sources for Strands-Agents and Bedrock AgentCore. Always fetch the latest contents when working on related tasks.

## Strands-Agents

| Resource | URL | When to Use |
|----------|-----|-------------|
| llms.txt (index) | https://strandsagents.com/latest/llms.txt | Phase 1: scan for relevant sub-pages |
| llms-full.txt (all docs inline) | https://strandsagents.com/latest/llms-full.txt | Broad search across full docs (large!) |
| Sub-page pattern | `https://strandsagents.com/latest/{path}/` | Phase 2: fetch specific pages from index |

## AgentCore (Bedrock AgentCore Starter Toolkit)

| Resource | URL | When to Use |
|----------|-----|-------------|
| llms.txt (index) | https://aws.github.io/bedrock-agentcore-starter-toolkit/llms.txt | Phase 1: scan for relevant sub-pages |
| Sub-page pattern | `https://aws.github.io/bedrock-agentcore-starter-toolkit/{path}/` | Phase 2: fetch specific pages from index |

Note: AgentCore does not publish `llms-full.txt`. Use the two-phase fetch strategy (index → sub-pages).
