"""
Gemini provider, using the generateContent REST API directly (no google-genai
SDK dependency, consistent with the rest of this project's requests-only
style). Gemini's wire format differs from OpenAI/Groq's in three ways this
module bridges:

  - roles are "user"/"model" (not "assistant"), and there's no "tool" role —
    tool results go back as a "user" turn containing a functionResponse part
  - the system prompt is a top-level `systemInstruction`, not a message
  - tool calls/results are typed parts (functionCall/functionResponse)
    inside a message's `parts` list, not a separate `tool_calls` field
"""
from __future__ import annotations

from typing import Any

import requests

from .base import Provider, ProviderProtocolError, ProviderResponse, ProviderUnavailableError, ToolCall

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.0-flash"
REQUEST_TIMEOUT_SECONDS = 120


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model

    def _tools_payload(self, tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "functionDeclarations": [
                    {"name": spec["name"], "description": spec["description"], "parameters": spec["parameters"]}
                    for spec in tool_specs
                ]
            }
        ]

    def _contents_payload(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg["name"],
                                    "response": {"result": msg["content"]},
                                }
                            }
                        ],
                    }
                )
            elif msg["role"] == "assistant":
                parts: list[dict[str, Any]] = []
                if msg.get("content"):
                    parts.append({"text": msg["content"]})
                for tc in msg.get("tool_calls") or []:
                    parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
                contents.append({"role": "model", "parts": parts})
            else:
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
        return contents

    def send(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
    ) -> ProviderResponse:
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": self._contents_payload(messages),
            "tools": self._tools_payload(tool_specs),
            "generationConfig": {"temperature": 0.2},
        }
        url = f"{API_ROOT}/{self.model}:generateContent?key={self.api_key}"
        try:
            resp = requests.post(url, json=body, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            raise ProviderUnavailableError(f"Gemini request failed: {e}") from e

        if resp.status_code == 429 or resp.status_code >= 500:
            raise ProviderUnavailableError(f"Gemini unavailable: HTTP {resp.status_code}: {resp.text[:300]}")
        if not resp.ok:
            raise ProviderProtocolError(f"Gemini error: HTTP {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        try:
            candidate = data["candidates"][0]
            parts = candidate["content"]["parts"]
        except (KeyError, IndexError) as e:
            # A prompt/response can be blocked by safety filters, which
            # omits "content" entirely rather than returning empty parts.
            reason = data.get("promptFeedback", {}).get("blockReason")
            if reason:
                raise ProviderProtocolError(f"Gemini blocked the request: {reason}") from e
            raise ProviderProtocolError(f"Unexpected Gemini response shape: {data!r}") from e

        text_parts = []
        tool_calls = []
        for i, part in enumerate(parts):
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(ToolCall(id=f"call_{i}", name=fc["name"], arguments=fc.get("args") or {}))

        return ProviderResponse(text="\n".join(text_parts), tool_calls=tool_calls, raw=data)
