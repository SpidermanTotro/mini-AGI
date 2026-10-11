"""Small, explicit local tool dispatcher for coding agents.

Only an allowlisted read-only workspace operation is available. Models cannot
execute shell commands, access network resources, or modify files through this
dispatcher. Tool arguments are checked before any filesystem access.
"""
from __future__ import annotations

import json
from typing import Any

from minagi.local_assistant_tools import LocalAssistantTools


class LocalToolDispatcher:
    def __init__(self, tools: LocalAssistantTools):
        self.tools = tools

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name != "read_workspace_file":
            return {"ok": False, "error": "tool not allowed"}
        if not isinstance(arguments, dict) or set(arguments) != {"relative_path"}:
            return {"ok": False, "error": "invalid arguments"}
        path = arguments["relative_path"]
        if not isinstance(path, str) or not path or len(path) > 4096:
            return {"ok": False, "error": "invalid path"}
        try:
            content = self.tools.read_workspace_file(path)
        except (OSError, UnicodeError, ValueError, PermissionError) as exc:
            return {"ok": False, "error": type(exc).__name__}
        return {"ok": True, "content": content}

    def dispatch_json(self, name: str, arguments_json: str) -> dict[str, Any]:
        if len(arguments_json) > 8192:
            return {"ok": False, "error": "arguments too large"}
        try:
            arguments = json.loads(arguments_json)
        except (json.JSONDecodeError, TypeError):
            return {"ok": False, "error": "invalid JSON"}
        return self.dispatch(name, arguments)
