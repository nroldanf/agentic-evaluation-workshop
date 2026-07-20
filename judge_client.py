"""Shared judge-model selection for LLM-as-a-Judge eval scripts.

Bedrock is the primary judge; a distinct local Ollama model is the fallback
if Bedrock is unreachable. The fallback is deliberately not the app's own
generator model (`OLLAMA_MODEL`), so judge != generator even during an
outage — avoids self-enhancement bias (see LIMITATIONS.md).
"""

import logging
import os

logger = logging.getLogger("judge_client")

BEDROCK_MODEL_ID = os.getenv("JUDGE_BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
FALLBACK_OLLAMA_MODEL = os.getenv("JUDGE_FALLBACK_OLLAMA_MODEL", "mistral:latest")


def invoke_judge(response_model, messages: list[dict]) -> tuple[object, str]:
    """Score with Bedrock; fall back to a local Ollama model if Bedrock fails.

    Returns the validated `response_model` instance and a string identifying
    which model actually produced it (e.g. "bedrock:...", "ollama:...") so
    callers can record it and avoid silently blending fallback-scored items
    with Bedrock-scored ones.
    """
    from langchain_aws import ChatBedrockConverse

    try:
        model = ChatBedrockConverse(model_id=BEDROCK_MODEL_ID, region_name=BEDROCK_REGION, temperature=0)
        result = model.with_structured_output(response_model).invoke(messages)
        return result, f"bedrock:{BEDROCK_MODEL_ID}"
    except Exception as exc:
        logger.warning(
            "Bedrock judge call failed (%s); falling back to local Ollama %s", exc, FALLBACK_OLLAMA_MODEL
        )
        from langchain_ollama import ChatOllama

        model = ChatOllama(model=FALLBACK_OLLAMA_MODEL, temperature=0)
        result = model.with_structured_output(response_model).invoke(messages)
        return result, f"ollama:{FALLBACK_OLLAMA_MODEL}"


def ensure_score_configs(client, configs: list[dict]) -> dict[str, str]:
    """Create the given Langfuse score configs if they don't already exist.

    Each entry in `configs` needs `name`, `data_type`, `description`, and
    (for NUMERIC) `min_value`/`max_value`. Returns {name: config_id}.
    """
    existing = {c.name: c.id for c in client.api.score_configs.get().data}
    config_ids = {}
    for cfg in configs:
        if cfg["name"] in existing:
            config_ids[cfg["name"]] = existing[cfg["name"]]
            continue
        created = client.api.score_configs.create(**cfg)
        config_ids[cfg["name"]] = created.id
        logger.info("Created score config %s (%s)", cfg["name"], created.id)
    return config_ids
