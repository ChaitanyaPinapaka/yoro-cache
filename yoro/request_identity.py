"""Canonical exact identity for requests, separate from semantic task matching."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

_AFFECTING_FIELDS = (
    "model", "response_format", "text", "reasoning", "stop", "seed", "top_p",
    "temperature", "max_tokens", "max_completion_tokens", "max_output_tokens",
    "frequency_penalty", "presence_penalty", "logit_bias", "n", "modalities",
    "audio", "tools", "functions", "tool_choice", "parallel_tool_calls",
    "previous_response_id", "conversation", "prompt", "include", "truncation",
)


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable(value).encode()).hexdigest()[:24]


def _last_user_index(messages: list) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "user":
            return i
    return None


def _context_messages(messages: list) -> list:
    """Return answer-affecting context, excluding the semantic task text itself."""
    last = _last_user_index(messages)
    out = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or i != last:
            out.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, list):
            non_text = [p for p in content if not (
                isinstance(p, dict) and p.get("type") in ("text", "input_text")
            )]
            if non_text:
                clone = {k: v for k, v in msg.items() if k != "content"}
                clone["content"] = non_text
                out.append(clone)
        extras = {k: v for k, v in msg.items() if k not in ("role", "content")}
        if extras:
            out.append({"role": "user", **extras})
    return out


def has_non_text_input(body: dict) -> bool:
    messages = body.get("messages", body.get("input", []))
    if isinstance(messages, str):
        return False
    for msg in messages if isinstance(messages, list) else []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict) and part.get("type") not in (
                "text", "input_text", "output_text"
            ):
                return True
    return False


@dataclass(frozen=True)
class RequestIdentity:
    signature: str
    scope: dict[str, str]


def request_identity(body: dict, *, workspace: str = "", operation: str = "chat") -> RequestIdentity:
    messages = body.get("messages", body.get("input", []))
    if isinstance(messages, str):
        messages = []
    envelope = {
        "operation": operation,
        "context": _context_messages(messages if isinstance(messages, list) else []),
        "parameters": {k: body[k] for k in _AFFECTING_FIELDS if k in body},
        "instructions": body.get("instructions"),
    }
    sig = _digest(envelope)
    scope = {"operation": operation, "model": str(body.get("model") or ""),
             "request_signature": sig}
    if workspace:
        scope["workspace"] = workspace
    return RequestIdentity(sig, scope)
