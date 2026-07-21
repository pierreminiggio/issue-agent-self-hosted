"""
Context compaction for provider requests.

The message history the orchestrator keeps (`Orchestrator.run`'s local
`messages` list) grows every round — every read_file/search_code result gets
appended, forever, for the life of one run. That's fine for GitHub comments
and replay (see history.py), but a free-tier provider's per-request or
per-minute token budget can be small enough (Groq's free tier: 12,000
tokens/minute) that a long-running conversation eventually exceeds it in a
single request, regardless of rate limiting.

This module produces a *separate, shrunk copy* of the message history for
sending to a provider — the original, full-detail list the orchestrator
keeps for comments/replay is never modified.

Strategy: keep full (capped-length) tool results for only the most recent
`keep_full_rounds` rounds; older rounds' tool results collapse to a short
placeholder. A "round" boundary is any assistant message — everything from
one assistant message up to (not including) the next is that round.
"""
from __future__ import annotations

from typing import Any

# Progressively more aggressive compaction levels, tried in order against
# the same provider before giving up on it and moving to the next one.
COMPACTION_LEVELS: list[dict[str, int]] = [
    {"keep_full_rounds": 6, "tool_result_chars": 3000},
    {"keep_full_rounds": 2, "tool_result_chars": 1200},
    {"keep_full_rounds": 0, "tool_result_chars": 400},
]


def compact_messages(
    messages: list[dict[str, Any]], keep_full_rounds: int, tool_result_chars: int
) -> list[dict[str, Any]]:
    round_start_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if len(round_start_indices) <= keep_full_rounds:
        cutoff_index = 0  # fewer rounds so far than the budget allows — keep everything
    else:
        cutoff_index = round_start_indices[-keep_full_rounds] if keep_full_rounds > 0 else len(messages)

    compacted: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "tool":
            compacted.append(msg)
            continue
        content = msg["content"]
        if i >= cutoff_index:
            if len(content) > tool_result_chars:
                content = content[:tool_result_chars] + f"... [truncated for context budget, {len(content)} chars total]"
            compacted.append({**msg, "content": content})
        else:
            compacted.append(
                {
                    **msg,
                    "content": (
                        f"[Older result from {msg.get('name', 'a tool call')} omitted to save context — "
                        "the full output is in this issue's comment history if you need it again; "
                        "re-call the tool if so.]"
                    ),
                }
            )
    return compacted
