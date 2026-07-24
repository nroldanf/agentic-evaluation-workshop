"""
Trajectory evaluation: compares the agent's actual tool-call trace against
the expected/unexpected calls defined for a golden case (see the Bilbo
case, designed specifically to test tool-use efficiency).

Two layers, following the BFCL (Berkeley Function Calling Leaderboard)
distinction between "did it call the right tool the right number of times"
and "did it pass reasonable arguments" -- call-count alone can hide a
correct-looking trace built on a bad query:

1. Call-count validation -- did it call each tool the right number of times?
2. Argument validation -- for each call, did it pass reasonable arguments,
   not just *some* argument? (cheap keyword-containment heuristic by
   default; swap in an LLM-judge for stricter checking, same pattern as
   the other eval_*.py modules.)

Functional style, no classes: eval_trajectory() returns a plain dict.
"""
from collections import Counter
from typing import Callable, Optional


def _default_arg_check(call_args: dict, expected_contains: list[str]) -> bool:
    """Cheap heuristic: check that the tool's arguments (flattened to text)
    contain at least one expected keyword. Good enough to unblock testing
    without an LLM call -- replace with a judge for stricter validation,
    e.g. checking semantic relevance rather than literal keyword overlap."""
    text_blob = " ".join(str(v) for v in call_args.values()).lower()
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
      "argument_mismatches": list,
      "manual_review_notes": list,
    }
    """
    arg_check_fn = arg_check_fn or _default_arg_check
    counts = Counter(call["tool"] for call in tool_call_trace)

    count_mismatches = []
    argument_mismatches = []

    for expected in golden.get("expected_tool_calls", []):
        tool = expected["tool"]
        expected_count = expected["expected_call_count"]
        actual_count = counts.get(tool, 0)
        if actual_count != expected_count:
            count_mismatches.append(
                f"{tool}: expected {expected_count} call(s), got {actual_count}"
            )

        expected_contains = expected.get("expected_query_contains")
        if expected_contains:
            matching_calls = [c for c in tool_call_trace if c["tool"] == tool]
            for call in matching_calls:
                if not arg_check_fn(call.get("args", {}), expected_contains):
                    argument_mismatches.append({
                        "tool": tool,
                        "args": call.get("args", {}),
                        "expected_keywords": expected_contains,
                    })

    # unexpected_tool_calls in our dataset are free-text notes (not always a
    # clean tool name), so these are surfaced for manual review rather than
    # auto-checked -- still worth having in the report.
    manual_notes = list(golden.get("unexpected_tool_calls", []))

    return {
        "tool_call_counts": dict(counts),
        "count_mismatches": count_mismatches,
        "argument_mismatches": argument_mismatches,
        "manual_review_notes": manual_notes,
    }
