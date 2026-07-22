"""
Provider-agnostic interface for a cloud reasoning model.

The orchestrator only ever talks to this interface, never to Groq's or
Gemini's SDK/wire format directly. That's what makes the failover in
orchestrator.py possible: if Groq is rate-limited, the exact same message
history is handed to the Gemini provider instead, unchanged.
"""
from __future__ import annotations

import re
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
    """Rate-limited, quota-exhausted, or transiently erroring. The
    orchestrator retries this same provider with backoff first (see
    orchestrator._try_provider); if retries are exhausted, it moves to the
    next configured provider.

    `retry_after_seconds`, when the provider's error told us exactly how
    long to wait, is honored instead of a guessed exponential backoff.
    """

    def __init__(self, message: str, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ProviderRequestTooLargeError(ProviderUnavailableError):
    """The request itself exceeded a per-request/per-minute token budget
    (e.g. Groq's HTTP 413 tokens-per-minute limit). Waiting doesn't fix
    this — shrinking the request does — so the orchestrator responds by
    retrying this provider with a smaller, more compacted message history
    rather than sleeping.
    """


class ProviderProtocolError(Exception):
    """The provider returned something we couldn't parse (malformed tool
    call args, unexpected response shape). Not necessarily retryable on a
    different provider, but distinct from ProviderUnavailableError so
    callers can decide.
    """


_RETRY_AFTER_PATTERNS = [
    re.compile(r"try again in\s*([\d.]+)\s*s", re.IGNORECASE),  # Groq
    re.compile(r'"retryDelay"\s*:\s*"([\d.]+)s"', re.IGNORECASE),  # Gemini
    re.compile(r"retry.{0,20}after\s*[:=]?\s*([\d.]+)", re.IGNORECASE),
]


def parse_retry_after_seconds(text: str) -> float | None:
    """Best-effort extraction of a provider-suggested wait time from an
    error body. Returns None if nothing recognizable is found, in which
    case the caller should fall back to exponential backoff."""
    for pattern in _RETRY_AFTER_PATTERNS:
        match = pattern.search(text or "")
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


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
