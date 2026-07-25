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
   `lookup_icd10` is called three times correctly, but none of those calls
   should ever be about PTSD, since the golden expects that diagnosis to be
   absent entirely.
4. Unexpected tool use -- did the agent call a tool that isn't part of the
   expected trajectory at all? (e.g. get_patient_history on a first
   encounter, where the golden's `expected_tool_calls` only mentions
   lookup_icd10)

Functional style, no classes: eval_trajectory() returns a plain dict.
"""
from collections import Counter
from typing import Callable, Optional


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
    tool_call_trace: list[dict],  # [{"tool": "lookup_icd10", "args": {"query": "..."}}, ...]
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