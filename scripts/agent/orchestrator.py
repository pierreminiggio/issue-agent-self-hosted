"""
The orchestrator is the "local agent" as a whole: it holds no model and does
no reasoning of its own. Each round it:

  1. sends the current message history + tool specs to a cloud provider
  2. if that provider is rate-limited or transiently erroring, waits and
     retries the SAME provider (honoring its suggested wait time when it
     gives one, otherwise exponential backoff) for up to MAX_RETRIES_PER_PROVIDER
     attempts before giving up on it and moving to the next configured one
  3. if a request is outright too large for the provider's budget, retries
     with a progressively smaller (compacted) message history instead of
     waiting — waiting doesn't shrink a request, compaction does
  4. executes whatever tool call(s) the model asked for, locally, sandboxed
     to the repo checkout (see tools.ToolExecutor)
  5. writes the round's full data to the transcript store (if configured)
     and posts a short preview + link as a GitHub comment, so a human can
     watch the conversation as it happens without hunting through logs
  6. feeds the tool results back in as the next turn, and repeats

The whole point of the retry logic is patience: rate limits on free-tier
providers are common and often self-clear within seconds to minutes, so the
agent should sit and wait rather than giving up — reserving provider
failover for genuine, sustained outages (e.g. a daily quota that won't
clear during this run).

It stops when the model calls `finish` (done) or `ask_user` (needs a human),
or when it runs out of iterations/time/providers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import context, history
from .providers.base import (
    Provider,
    ProviderProtocolError,
    ProviderRequestTooLargeError,
    ProviderUnavailableError,
    ToolCall,
)
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

# How many times to retry the SAME provider on a rate limit / transient error
# before giving up on it and moving to the next configured provider. High on
# purpose: "slow and steady" — a free-tier rate limit is usually seconds to
# low minutes, not a sustained outage, so it's almost always worth waiting.
MAX_RETRIES_PER_PROVIDER = 20
BACKOFF_BASE_SECONDS = 5.0
BACKOFF_MAX_SECONDS = 90.0
# Small safety margin added on top of a provider's own suggested wait time.
RETRY_AFTER_BUFFER_SECONDS = 1.0


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
        transcript_store=None,  # transcript_store.TranscriptStore | None
        max_rounds: int = 40,
        max_seconds: int = 4 * 3600,
    ):
        if not providers:
            raise ValueError("At least one provider is required")
        self.providers = providers
        self.executor = executor
        self.post_comment = post_comment
        self.transcript_store = transcript_store
        self.max_rounds = max_rounds
        self.max_seconds = max_seconds

    def _try_provider(self, provider: Provider, system_prompt: str, messages: list[dict[str, Any]], deadline: float):
        """Repeatedly attempts one provider: shrinks the request on a
        too-large error, waits (provider-suggested time if given, else
        exponential backoff) and retries on rate limit/transient errors, up
        to MAX_RETRIES_PER_PROVIDER attempts total. Raises the last error if
        it never succeeds — the caller (_send_with_failover) then moves on
        to the next configured provider.
        """
        level_index = 0
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES_PER_PROVIDER + 1):
            level = context.COMPACTION_LEVELS[min(level_index, len(context.COMPACTION_LEVELS) - 1)]
            trimmed = context.compact_messages(messages, **level)
            try:
                return provider.send(system_prompt, trimmed, TOOL_SPECS)
            except ProviderRequestTooLargeError as e:
                last_error = e
                level_index += 1  # shrink further next attempt; no need to wait, size is the problem
                print(f"{provider.name}: request too large (attempt {attempt}), shrinking and retrying: {e}")
                continue
            except ProviderUnavailableError as e:
                last_error = e
                remaining = deadline - time.monotonic()
                if remaining <= 1:
                    print(f"{provider.name}: out of time budget, not waiting further: {e}")
                    raise
                if e.retry_after_seconds is not None:
                    wait = e.retry_after_seconds + RETRY_AFTER_BUFFER_SECONDS
                else:
                    wait = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
                wait = min(wait, max(remaining - 1, 1))
                print(
                    f"{provider.name}: unavailable (attempt {attempt}/{MAX_RETRIES_PER_PROVIDER}), "
                    f"waiting {wait:.1f}s before retrying: {e}"
                )
                time.sleep(wait)
                continue
            except ProviderProtocolError:
                raise  # not something waiting or shrinking fixes — bubble up immediately

        raise last_error or RuntimeError(f"{provider.name}: exhausted all retries")

    def _send_with_failover(self, system_prompt: str, messages: list[dict[str, Any]], deadline: float):
        errors = []
        for provider in self.providers:
            try:
                return provider, self._try_provider(provider, system_prompt, messages, deadline)
            except (ProviderUnavailableError, ProviderProtocolError) as e:
                errors.append(f"{provider.name}: {e}")
                continue
        raise RuntimeError(
            "All configured providers failed or are unavailable this round:\n" + "\n".join(errors)
        )

    def _post_round(self, issue_number: int, round_number: int, provider_name: str, assistant_text: str,
                     tool_calls: list[ToolCall], tool_results: list[dict[str, str]]) -> None:
        store_ref = store_path = store_url = None
        if self.transcript_store is not None and (tool_calls or tool_results):
            try:
                full_payload = {
                    "round": round_number,
                    "provider": provider_name,
                    "assistant_text": assistant_text,
                    "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments or {}} for tc in tool_calls],
                    "tool_results": tool_results,
                }
                store_path, store_url = self.transcript_store.save_round(issue_number, round_number, full_payload)
                store_ref = self.transcript_store.branch
            except Exception as e:  # noqa: BLE001 — the store is a nice-to-have, not required to proceed
                print(f"WARNING: failed to write full transcript to store, falling back to embedding in comment: {e}")

        self.post_comment(
            history.render_transcript_comment(
                round_number, provider_name, assistant_text, tool_calls, tool_results,
                store_ref=store_ref, store_path=store_path, store_url=store_url,
            )
        )

    def run(self, repo: str, issue_number: int, initial_messages: list[dict[str, Any]]) -> RunResult:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(issue_number=issue_number, repo=repo)
        messages: list[dict[str, Any]] = list(initial_messages)
        start = time.monotonic()
        deadline = start + self.max_seconds
        text_only_streak = 0

        for round_number in range(1, self.max_rounds + 1):
            if time.monotonic() >= deadline:
                return RunResult(status="error", error="Time budget exhausted.", rounds_used=round_number - 1)

            try:
                provider, response = self._send_with_failover(system_prompt, messages, deadline)
            except RuntimeError as e:
                return RunResult(status="error", error=str(e), rounds_used=round_number - 1)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.text}
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)

            if not response.tool_calls:
                text_only_streak += 1
                self._post_round(issue_number, round_number, provider.name, response.text, [], [])
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

            self._post_round(issue_number, round_number, provider.name, response.text, response.tool_calls, tool_results)
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
