"""
Trajectory evaluation: compares the agent's actual tool-call trace against
the expected calls defined for a golden case (see the Bilbo case, designed
specifically to test tool-use efficiency).

The golden's `expected_tool_calls` lists only the tools that SHOULD be used --
it is the ideal trajectory, not an exhaustive list of do's and don'ts. Any
tool the agent calls that isn't in that list at all is automatically
flagged, the same way eval_diagnoses flags a code that's in the extraction
but not in the golden: absence from the golden is itself the signal.

Four things get checked:

1. Call-count validation -- did it call each *expected* tool the right
   number of times?
2. Required-content validation -- for a tool that legitimately gets called,
   did at least the expected content show up in its arguments?
   (`expected_query_contains`)
3. Forbidden-content validation -- for a tool that legitimately gets called
   multiple times, did any *specific* call search for something it
   shouldn't have? (`forbidden_query_contains`) -- e.g. Houses of Healing:
   `search_icd10_codes_batch` is called three times correctly, but none of
   those calls should ever be about PTSD, since the golden expects that
   diagnosis to be absent entirely.
4. Unexpected tool use -- did the agent call a tool that isn't part of the
   expected trajectory at all? (e.g. save_json getting called mid-extraction,
   where the golden's `expected_tool_calls` only mentions
   search_icd10_codes_batch)

Functional style, no classes: eval_trajectory() returns a plain dict.
"""
import logging
from collections import Counter
from typing import Callable, Optional

logger = logging.getLogger("eval_trajectory")

# Score configs for log_trajectory_scores(). Pass through
# judge_client.ensure_score_configs(client, SCORE_CONFIGS) once per process to
# get the {name: config_id} map that function expects.
SCORE_CONFIGS = [
    {
        "name": "trajectory_call_counts_match",
        "data_type": "BOOLEAN",
        "description": "Whether every expected tool was called exactly the expected number of times.",
    },
    {
        "name": "trajectory_arguments_ok",
        "data_type": "BOOLEAN",
        "description": "Whether expected tool calls' arguments contained the required content.",
    },
    {
        "name": "trajectory_no_forbidden_queries",
        "data_type": "BOOLEAN",
        "description": "Whether any tool call's arguments matched content that golden marks as forbidden.",
    },
    {
        "name": "trajectory_no_unexpected_tools",
        "data_type": "BOOLEAN",
        "description": "Whether the agent called only tools present in the golden's expected_tool_calls.",
    },
]


def _text_blob(call_args: dict) -> str:
    return " ".join(str(v) for v in call_args.values()).lower()


def _default_arg_check(call_args: dict, expected_contains: list[str]) -> bool:
    """Cheap heuristic: check that the tool's arguments (flattened to text)
    contain at least one expected keyword. Good enough to unblock testing
    without an LLM call -- replace with a judge for stricter validation,
    e.g. checking semantic relevance rather than literal keyword overlap."""
    text_blob = _text_blob(call_args)
    return any(kw.lower() in text_blob for kw in expected_contains)


def eval_trajectory(
    golden: dict,
    tool_call_trace: list[dict],  # [{"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["..."]}}, ...]
    arg_check_fn: Optional[Callable[[dict, list[str]], bool]] = None,
) -> dict:
    """Returns a plain dict:
    {
      "tool_call_counts": dict,
      "count_mismatches": list,
      "argument_mismatches": list,        # missing expected content
      "forbidden_query_matches": list,    # hit content that should never appear
      "unexpected_tools_called": list,    # tools called that aren't in the golden at all
    }
    """
    arg_check_fn = arg_check_fn or _default_arg_check
    counts = Counter(call["tool"] for call in tool_call_trace)

    expected_by_tool = {e["tool"]: e for e in golden.get("expected_tool_calls", [])}

    count_mismatches = []
    argument_mismatches = []
    forbidden_query_matches = []

    for tool, expected in expected_by_tool.items():
        expected_count = expected["expected_call_count"]
        actual_count = counts.get(tool, 0)
        if actual_count != expected_count:
            count_mismatches.append(
                f"{tool}: expected {expected_count} call(s), got {actual_count}"
            )

        matching_calls = [c for c in tool_call_trace if c["tool"] == tool]

        expected_contains = expected.get("expected_query_contains")
        if expected_contains:
            for call in matching_calls:
                if not arg_check_fn(call.get("args", {}), expected_contains):
                    argument_mismatches.append({
                        "tool": tool,
                        "args": call.get("args", {}),
                        "expected_keywords": expected_contains,
                    })

        forbidden_contains = expected.get("forbidden_query_contains")
        if forbidden_contains:
            for call in matching_calls:
                blob = _text_blob(call.get("args", {}))
                hit = [kw for kw in forbidden_contains if kw.lower() in blob]
                if hit:
                    forbidden_query_matches.append({
                        "tool": tool,
                        "args": call.get("args", {}),
                        "matched_forbidden_terms": hit,
                    })

    unexpected_tools_called = [
        {"tool": tool_name, "call_count": count}
        for tool_name, count in counts.items()
        if tool_name not in expected_by_tool
    ]

    return {
        "tool_call_counts": dict(counts),
        "count_mismatches": count_mismatches,
        "argument_mismatches": argument_mismatches,
        "forbidden_query_matches": forbidden_query_matches,
        "unexpected_tools_called": unexpected_tools_called,
    }


def log_trajectory_scores(
    client,
    trace_id: str,
    result: dict,
    observation_id: Optional[str] = None,
    config_ids: Optional[dict[str, str]] = None,
) -> None:
    """Attach eval_trajectory()'s metrics to a Langfuse trace (optionally
    scoped to one observation) as BOOLEAN scores. `config_ids` is the
    {name: config_id} map from judge_client.ensure_score_configs(client, SCORE_CONFIGS)."""
    config_ids = config_ids or {}
    scores = [
        ("trajectory_call_counts_match", not result["count_mismatches"]),
        ("trajectory_arguments_ok", not result["argument_mismatches"]),
        ("trajectory_no_forbidden_queries", not result["forbidden_query_matches"]),
        ("trajectory_no_unexpected_tools", not result["unexpected_tools_called"]),
    ]
    for name, value in scores:
        client.create_score(
            name=name,
            value=int(value),
            trace_id=trace_id,
            observation_id=observation_id,
            data_type="BOOLEAN",
            config_id=config_ids.get(name),
        )
    if result["count_mismatches"]:
        logger.warning("trace %s: tool call count mismatches: %s", trace_id, result["count_mismatches"])
    if result["argument_mismatches"]:
        logger.warning("trace %s: %d tool argument mismatch(es)", trace_id, len(result["argument_mismatches"]))
    if result["forbidden_query_matches"]:
        logger.warning("trace %s: forbidden query match(es): %s", trace_id, result["forbidden_query_matches"])
    if result["unexpected_tools_called"]:
        logger.warning("trace %s: unexpected tools called: %s", trace_id, result["unexpected_tools_called"])


def run_self_tests() -> list[str]:
    """Fabricated single-error tool-trace scenarios built directly from
    data/golden/golden_encounter_{riv001,hou002}.json's own expected_tool_calls
    -- no invented patient, no other golden file. Each scenario changes
    exactly one thing about the trace, then asserts eval_trajectory actually
    catches it. Returns the list of failed scenario labels (empty if
    everything passed). Run via `uv run python evals/eval_trajectory.py`.
    """
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    golden = {
        name: json.loads((repo_root / f"data/golden/golden_encounter_{name}.json").read_text(encoding="utf-8"))
        for name in ("riv001", "hou002")
    }

    failed = []

    def check(label: str, condition: bool, detail="") -> None:
        if condition:
            print(f"  PASS  {label}")
        else:
            failed.append(label)
            print(f"  FAIL  {label}  {detail}")

    print("=== Trajectory (eval_trajectory) ===")

    g = golden["riv001"]  # expected: search_icd10_codes_batch x3, no forbidden terms defined

    trace = [
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["puncture wound shoulder"]}},
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["sepsis toxic wound"]}},
    ]
    result = eval_trajectory(g, trace)
    check(
        "tool_call_count_too_few -> count_mismatches non-empty, nothing else",
        bool(result["count_mismatches"]) and not result["argument_mismatches"] and not result["unexpected_tools_called"],
        result,
    )

    trace = [
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["puncture wound shoulder"]}},
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["sepsis toxic wound"]}},
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["shoulder infection"]}},
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["wound care"]}},
    ]
    result = eval_trajectory(g, trace)
    check("tool_call_count_too_many -> count_mismatches non-empty", bool(result["count_mismatches"]), result)

    trace = [
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["puncture wound shoulder"]}},
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["sepsis toxic wound"]}},
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["unrelated headache"]}},
    ]
    result = eval_trajectory(g, trace)
    check(
        "argument_mismatch -> flagged, count is correct",
        bool(result["argument_mismatches"]) and not result["count_mismatches"],
        result,
    )

    # HOU-002 golden explicitly forbids PTSD-related queries.
    g2 = golden["hou002"]
    trace = [
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["dehydration"]}},
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["amputation finger"]}},
        # Includes "stress" (an expected keyword) alongside "PTSD" (forbidden),
        # isolating the forbidden-term check from the argument-match check.
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["stress reaction, possible PTSD"]}},
    ]
    result = eval_trajectory(g2, trace)
    check(
        "forbidden_query_match -> flagged, argument check still satisfied",
        bool(result["forbidden_query_matches"]) and not result["argument_mismatches"],
        result,
    )

    trace = [
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["puncture wound shoulder"]}},
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["sepsis toxic wound"]}},
        {"tool": "search_icd10_codes_batch", "args": {"diagnosis_names": ["shoulder infection"]}},
        {"tool": "save_json", "args": {"filename": "note.json", "data": {}}},
    ]
    result = eval_trajectory(g, trace)
    check(
        "unexpected_tool_called -> flagged, everything else clean",
        bool(result["unexpected_tools_called"]) and not result["count_mismatches"] and not result["argument_mismatches"],
        result,
    )

    print(f"\n{len(failed)} failed" if failed else "\nall passed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(1 if run_self_tests() else 0)