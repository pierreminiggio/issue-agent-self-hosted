"""
Groq provider. Groq exposes an OpenAI-compatible /chat/completions endpoint,
so this is a fairly direct translation of the shared message format into
that wire format and back.
"""
from __future__ import annotations

import json
from typing import Any

import requests

from .base import (
    Provider,
    ProviderProtocolError,
    ProviderRequestTooLargeError,
    ProviderResponse,
    ProviderUnavailableError,
    ToolCall,
)

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT_SECONDS = 120


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model

    def _tools_payload(self, tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
            for spec in tool_specs
        ]

    def _messages_payload(self, system_prompt: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            if msg["role"] == "tool":
                payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg["tool_call_id"],
                        "content": msg["content"],
                    }
                )
            elif msg["role"] == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or None}
                if msg.get("tool_calls"):
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in msg["tool_calls"]
                    ]
                payload.append(entry)
            else:
                payload.append({"role": "user", "content": msg["content"]})
        return payload

    def send(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
    ) -> ProviderResponse:
        body = {
            "model": self.model,
            "messages": self._messages_payload(system_prompt, messages),
            "tools": self._tools_payload(tool_specs),
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as e:
            raise ProviderUnavailableError(f"Groq request failed: {e}") from e

        if resp.status_code == 413:
            raise ProviderRequestTooLargeError(f"Groq request too large: HTTP 413: {resp.text[:300]}")
        if resp.status_code == 429 or resp.status_code >= 500:
            raise ProviderUnavailableError(f"Groq unavailable: HTTP {resp.status_code}: {resp.text[:300]}")
        if not resp.ok:
            raise ProviderProtocolError(f"Groq error: HTTP {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        try:
            choice = data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise ProviderProtocolError(f"Unexpected Groq response shape: {data!r}") from e

        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                # Some models emit a literal "null" (valid JSON, but not a
                # dict) for tool calls that take no parameters.
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))

        return ProviderResponse(text=choice.get("content") or "", tool_calls=tool_calls, raw=data)
