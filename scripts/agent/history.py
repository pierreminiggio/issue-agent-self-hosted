"""
"Trust the messenger": the sole defense against a prompt-injected issue or
comment is checking WHO posted it before anything about WHAT it says is
read. Exactly one GitHub login is trusted — the account that owns the PAT
this bot posts with (see GitHubClient.get_authenticated_login). Anything
authored by a different login is dropped before its .body is ever looked
at, logged, or handed to a model.

This module also owns the transcript comment format: every round of the
local-agent/cloud-model exchange is posted to the issue as one comment, in
two parts — a JSON blob (for a future run to parse back into exact replay
history) and a human-readable rendition underneath it (for you to read in
the GitHub UI). Only comments authored by the trusted login and carrying
these markers are ever parsed back as structured history; everything else
authored by that same login is treated as plain human feedback text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .providers.base import ToolCall

TRANSCRIPT_MARKER = "issue-agent:transcript"
QUESTION_MARKER = "issue-agent:question"
REFUSAL_MARKER = "issue-agent:refused"

# Tool results can be large (up to repo_tools.MAX_READ_CHARS per read); cap
# what we store/replay in a comment well below GitHub's ~65k comment body
# limit and well below what's pleasant to scroll past. The model can always
# re-call a tool if it needs the untruncated content again later.
TOOL_RESULT_STORE_CHARS = 1500

_BLOCK_RE = re.compile(
    r"<!--\s*(?P<marker>issue-agent:\w+)\s*(?P<json>\{.*?\})?\s*-->", re.DOTALL
)


class UntrustedIssueError(Exception):
    """The issue was not authored by the trusted login. Per policy, the run
    must refuse entirely rather than proceed with partial trust."""


@dataclass
class TrustedComment:
    kind: str  # "transcript" | "question" | "human"
    created_at: str
    data: dict[str, Any] | None  # parsed JSON payload, for "transcript"/"question"
    body: str | None  # raw text, for "human" only — never set for the other kinds


def filter_trusted(issue: dict, trusted_login: str) -> list[TrustedComment]:
    """Checks the issue author, then classifies each comment by author +
    marker. Raises UntrustedIssueError if the issue itself isn't from the
    trusted login — checked first, before anything else, and before any
    comment is even looked at.
    """
    issue_author = (issue.get("user") or {}).get("login")
    if issue_author != trusted_login:
        raise UntrustedIssueError(
            f"Issue #{issue.get('number')} was opened by '{issue_author}', not the trusted "
            f"account '{trusted_login}'. Refusing to run."
        )

    trusted: list[TrustedComment] = []
    for comment in issue.get("comments_data", []):
        author = (comment.get("user") or {}).get("login")
        if author != trusted_login:
            continue  # discarded here; comment["body"] is never read past this line
        trusted.append(_classify(comment))
    return trusted


def _classify(comment: dict) -> TrustedComment:
    body = comment.get("body") or ""
    match = _BLOCK_RE.search(body)
    created_at = comment.get("created_at", "")
    if match and match.group("marker") == TRANSCRIPT_MARKER and match.group("json"):
        try:
            return TrustedComment("transcript", created_at, json.loads(match.group("json")), None)
        except json.JSONDecodeError:
            pass  # fall through to treating it as opaque human text
    if match and match.group("marker") == QUESTION_MARKER:
        return TrustedComment("question", created_at, {}, None)
    return TrustedComment("human", created_at, None, body)


def build_initial_messages(issue: dict, trusted_comments: list[TrustedComment]) -> list[dict[str, Any]]:
    """Reconstructs the provider-agnostic message history from the trusted
    issue body plus every trusted comment, in chronological order. Used both
    for a brand-new run (no transcript comments yet, just the issue body)
    and for a resumed run continuing a previous conversation (e.g. after an
    ask_user pause, or after a run stopped mid-way).
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": f"Issue title: {issue.get('title', '')}\n\n{issue.get('body') or '(no description provided)'}",
        }
    ]

    for item in trusted_comments:
        if item.kind == "transcript":
            data = item.data or {}
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("arguments") or {})
                for tc in data.get("tool_calls", [])
            ]
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": data.get("assistant_text") or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            for tr in data.get("tool_results", []):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "name": tr["name"],
                        "content": tr["content"],
                    }
                )
        elif item.kind == "human":
            messages.append({"role": "user", "content": item.body})
        # "question" comments carry no extra content beyond what the round's
        # own tool_results already captured (the ask_user call/result) —
        # nothing further to replay.

    return messages


def render_transcript_comment(
    round_number: int,
    provider_name: str,
    assistant_text: str,
    tool_calls: list[ToolCall],
    tool_results: list[dict[str, str]],
) -> str:
    stored_results = [
        {
            "tool_call_id": tr["tool_call_id"],
            "name": tr["name"],
            "content": _truncate(tr["content"], TOOL_RESULT_STORE_CHARS),
        }
        for tr in tool_results
    ]
    payload = {
        "round": round_number,
        "provider": provider_name,
        "assistant_text": assistant_text,
        "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments or {}} for tc in tool_calls],
        "tool_results": stored_results,
    }
    json_block = f"<!-- {TRANSCRIPT_MARKER}\n{json.dumps(payload)}\n-->"

    lines = [json_block, "", f"**🤖 Round {round_number} — via {provider_name}**"]
    if assistant_text.strip():
        quoted = "\n".join(f"> {line}" for line in assistant_text.strip().splitlines())
        lines += ["", quoted]
    if tool_calls:
        lines.append("")
        lines.append("**Tool calls:**")
        by_id = {tr["tool_call_id"]: tr for tr in tool_results}
        for tc in tool_calls:
            args_str = ", ".join(f"{k}={v!r}" for k, v in (tc.arguments or {}).items())
            result = by_id.get(tc.id, {}).get("content", "")
            result_preview = _truncate(result, 300).replace("\n", " ")
            lines.append(f"- `{tc.name}({args_str})` → {result_preview}")
    return "\n".join(lines)


def render_question_comment(question: str) -> str:
    marker = f"<!-- {QUESTION_MARKER} -->"
    return (
        f"{marker}\n\n"
        f"**🤖 The agent has a question and is pausing here:**\n\n"
        f"> {question}\n\n"
        f"Reply on this issue (from the same account) to continue, then re-run the workflow."
    )


def render_refusal_comment(trusted_login: str) -> str:
    marker = f"<!-- {REFUSAL_MARKER} -->"
    return (
        f"{marker}\n\n"
        f"**⚠️ Run refused.** This issue was not opened by the account this agent trusts "
        f"(`{trusted_login}`). To avoid acting on instructions from an untrusted source, no "
        f"content from this issue was read. Re-open the issue from `{trusted_login}` to proceed."
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text)} chars total]"
