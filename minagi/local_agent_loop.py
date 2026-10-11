"""Bounded local model tool-call loop using an injected model callback.

The callback is responsible for contacting the chosen local model. This module
never makes network calls and never grants filesystem permission itself.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from minagi.local_tool_dispatcher import LocalToolDispatcher

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_workspace_file",
        "description": "Read a UTF-8 file within the explicitly permitted workspace.",
        "parameters": {
            "type": "object",
            "properties": {"relative_path": {"type": "string"}},
            "required": ["relative_path"],
            "additionalProperties": False,
        },
    },
}


def run_local_agent(
    model_reply: Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]],
    messages: list[dict[str, Any]],
    dispatcher: LocalToolDispatcher,
    *,
    max_tool_calls: int = 4,
) -> tuple[str, list[dict[str, Any]]]:
    """Return final text and transcript; fail closed on malformed tool calls."""
    if not 0 <= max_tool_calls <= 20:
        raise ValueError("max_tool_calls must be between 0 and 20")
    transcript = [dict(message) for message in messages]
    calls_used = 0
    for _ in range(max_tool_calls + 2):
        reply = model_reply(transcript, [READ_FILE_TOOL])
        if not isinstance(reply, dict) or reply.get("role") != "assistant":
            raise ValueError("model returned an invalid assistant message")
        raw_calls = reply.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ValueError("model returned invalid tool calls")
        if not raw_calls:
            content = reply.get("content")
            if not isinstance(content, str):
                raise ValueError("model returned no text")
            transcript.append({"role": "assistant", "content": content})
            return content, transcript
        if len(raw_calls) > max_tool_calls - calls_used:
            raise RuntimeError("local tool call limit exceeded")
        validated = []
        for call in raw_calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise ValueError("malformed tool call")
            fn = call["function"]
            name = fn.get("name")
            args = fn.get("arguments")
            if not isinstance(name, str) or not isinstance(args, (dict, str)):
                raise ValueError("malformed tool call arguments")
            validated.append((name, args))
        transcript.append(reply)
        for name, args in validated:
            result = (dispatcher.dispatch_json(name, args) if isinstance(args, str)
                      else dispatcher.dispatch(name, args))
            transcript.append({"role": "tool", "name": name,
                               "content": json.dumps(result)})
            calls_used += 1
    raise RuntimeError("local agent did not finish within its bounded tool loop")
