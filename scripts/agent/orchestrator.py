"""
The orchestrator is the "local agent" as a whole: it holds no model and does
no reasoning of its own. Each round it:

  1. sends the current message history + tool specs to a cloud provider
  2. if that provider is unavailable (rate limit / outage), retries the same
     request on the next configured provider, unchanged
  3. executes whatever tool call(s) the model asked for, locally, sandboxed
     to the repo checkout (see tools.ToolExecutor)
  4. posts the round (model's text + tool calls + tool results) as one
     GitHub comment, so a human can watch the conversation as it happens
  5. feeds the tool results back in as the next turn, and repeats

It stops when the model calls `finish` (done) or `ask_user` (needs a human),
or when it runs out of iterations/time/providers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import history
from .providers.base import Provider, ProviderProtocolError, ProviderUnavailableError, ToolCall
from .tools import TOOL_SPECS, ToolExecutionError, ToolExecutor

SYSTEM_PROMPT_TEMPLATE = """\
You are an autonomous software engineer working against a real Git repository \
checkout, resolving GitHub issue #{issue_number} in {repo}.

You do not have the repository contents yet. You have a small set of tools to \
explore and modify it — you must call these tools to get anything done; you \
cannot see or edit files any other way.

Rules:
- Always call get_project_tree first if you haven't already, to orient yourself.
- Read the actual files you plan to change before editing them; do not guess \
at their contents.
- Prefer edit_file (exact old_str/new_str replacement) over write_file for any \
file that already exists. write_file only works for brand-new files.
- If the repository has a test suite, run it with run_tests after making \
changes, and fix any failures before finishing.
- If you're missing information you can't reasonably infer from the code \
(business rules, product decisions, ambiguous requirements), call ask_user \
instead of guessing. This ends the run until a human replies, so use it only \
when truly needed — inspect the code first.
- When the feature/fix is complete (and tests pass, if any exist), call \
finish with a clear summary of what you changed and why.
- You must call exactly one tool every turn. Do not respond with plain text only.
"""

MAX_ROUNDS_WITHOUT_TOOL_CALL = 2


@dataclass
class RunResult:
    status: str  # "finished" | "waiting_for_human" | "error"
    summary: str | None = None
    question: str | None = None
    error: str | None = None
    rounds_used: int = 0


@dataclass
class _Round:
    provider_name: str
    assistant_text: str
    tool_calls: list[ToolCall]
    tool_results: list[dict[str, str]]


class Orchestrator:
    def __init__(
        self,
        providers: list[Provider],
        executor: ToolExecutor,
        post_comment,  # Callable[[str], None] — posts a rendered comment body
        max_rounds: int = 40,
        max_seconds: int = 4 * 3600,
    ):
        if not providers:
            raise ValueError("At least one provider is required")
        self.providers = providers
        self.executor = executor
        self.post_comment = post_comment
        self.max_rounds = max_rounds
        self.max_seconds = max_seconds

    def _send_with_failover(self, system_prompt: str, messages: list[dict[str, Any]]):
        errors = []
        for provider in self.providers:
            try:
                return provider, provider.send(system_prompt, messages, TOOL_SPECS)
            except ProviderUnavailableError as e:
                errors.append(f"{provider.name}: {e}")
                continue
            except ProviderProtocolError as e:
                errors.append(f"{provider.name}: {e}")
                continue
        raise RuntimeError(
            "All configured providers failed or are unavailable this round:\n" + "\n".join(errors)
        )

    def run(self, repo: str, issue_number: int, initial_messages: list[dict[str, Any]]) -> RunResult:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(issue_number=issue_number, repo=repo)
        messages: list[dict[str, Any]] = list(initial_messages)
        start = time.monotonic()
        text_only_streak = 0

        for round_number in range(1, self.max_rounds + 1):
            if time.monotonic() - start > self.max_seconds:
                return RunResult(status="error", error="Time budget exhausted.", rounds_used=round_number - 1)

            try:
                provider, response = self._send_with_failover(system_prompt, messages)
            except RuntimeError as e:
                return RunResult(status="error", error=str(e), rounds_used=round_number - 1)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.text}
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)

            if not response.tool_calls:
                text_only_streak += 1
                self.post_comment(
                    history.render_transcript_comment(round_number, provider.name, response.text, [], [])
                )
                if text_only_streak >= MAX_ROUNDS_WITHOUT_TOOL_CALL:
                    return RunResult(
                        status="error",
                        error="Model repeatedly responded without calling a tool.",
                        rounds_used=round_number,
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You must call exactly one tool to proceed (use `finish` if you're "
                            "done, or `ask_user` if you need clarification)."
                        ),
                    }
                )
                continue

            text_only_streak = 0
            tool_results = []
            terminal: tuple[str, dict] | None = None

            def _on_terminal(kind: str, args: dict) -> None:
                nonlocal terminal
                terminal = (kind, args)

            executor = self.executor
            executor.on_terminal = _on_terminal

            for tc in response.tool_calls:
                try:
                    result_text = executor.execute(tc.name, tc.arguments)
                except ToolExecutionError as e:
                    result_text = f"ERROR: {e}"
                tool_results.append({"tool_call_id": tc.id, "name": tc.name, "content": result_text})

            self.post_comment(
                history.render_transcript_comment(
                    round_number, provider.name, response.text, response.tool_calls, tool_results
                )
            )
            for tr in tool_results:
                messages.append(
                    {"role": "tool", "tool_call_id": tr["tool_call_id"], "name": tr["name"], "content": tr["content"]}
                )

            if terminal is not None:
                kind, args = terminal
                if kind == "ask_user":
                    question = executor.pending_question or "(no question text provided)"
                    self.post_comment(history.render_question_comment(question))
                    return RunResult(status="waiting_for_human", question=question, rounds_used=round_number)
                if kind == "finish":
                    return RunResult(
                        status="finished", summary=executor.finish_summary, rounds_used=round_number
                    )

        return RunResult(status="error", error=f"Ran out of iterations ({self.max_rounds}).", rounds_used=self.max_rounds)
