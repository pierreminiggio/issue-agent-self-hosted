"""
Provider-agnostic interface for a cloud reasoning model.

The orchestrator only ever talks to this interface, never to Groq's or
Gemini's SDK/wire format directly. That's what makes the failover in
orchestrator.py possible: if Groq is rate-limited, the exact same message
history is handed to the Gemini provider instead, unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderResponse:
    # Free-text the model produced this turn (its "reasoning out loud" /
    # commentary). May be empty if the model only made tool calls.
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Raw provider payload, kept only for debugging/logging, never parsed by
    # the orchestrator.
    raw: Any = None


class ProviderUnavailableError(Exception):
    """Rate-limited, quota-exhausted, or transiently erroring — the
    orchestrator should retry this same request on the *other* provider
    rather than fail the whole run.
    """


class ProviderProtocolError(Exception):
    """The provider returned something we couldn't parse (malformed tool
    call args, unexpected response shape). Not necessarily retryable on a
    different provider, but distinct from ProviderUnavailableError so
    callers can decide.
    """


class Provider:
    name: str = "base"

    def send(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
    ) -> ProviderResponse:
        """`messages` is a provider-agnostic list of
        {"role": "user"|"assistant"|"tool", "content": str, ...} dicts as
        built by orchestrator.Conversation. Each provider is responsible for
        translating this (and `tool_specs`) into its own wire format and
        translating the response back into a ProviderResponse.
        """
        raise NotImplementedError
