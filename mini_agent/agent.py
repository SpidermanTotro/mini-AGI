"""Tool-calling assistant with local SQLite conversation history."""

import json
import sqlite3
from pathlib import Path

from .tools import TOOL_SPECS, WorkspaceTools


SYSTEM_PROMPT = """You are a local assistant operating inside one user-selected workspace.
Be honest about uncertainty. Use available tools when they help. Workspace reads
are limited to the selected root. File writes require the user's approval.
Running project tests executes project code and also requires approval. Python
syntax checks only compile source and do not execute it. There is no general
shell-command tool. For programming tasks, inspect relevant files first, make
changes only through approved workspace writes, check Python syntax, and ask
before running tests. For image requests, use the image tool only when asked;
image synthesis is performed by a separate configured local diffusion server,
not by the byte-level mini-AGI model.
Never claim you changed or checked something unless a tool result confirms it.
"""


class LocalAgent:
    def __init__(self, client, workspace, database, confirm_write=None,
                 confirm_run=None, history_turns=8, max_tool_rounds=6,
                 image_api_url=None):
        self.client = client
        self.tools = WorkspaceTools(workspace, confirm_write=confirm_write,
                                    confirm_run=confirm_run,
                                    image_api_url=image_api_url)
        self.history_turns = history_turns
        self.max_tool_rounds = max_tool_rounds
        database = str(database)
        if database != ":memory:":
            Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)
            database = str(Path(database).expanduser())
        self.db = sqlite3.connect(database)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "conversation TEXT NOT NULL, turn INTEGER NOT NULL, "
            "seq INTEGER NOT NULL, body TEXT NOT NULL, "
            "PRIMARY KEY (conversation, turn, seq))")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS messages_conversation_turn "
            "ON messages(conversation, turn)")
        self.db.commit()

    def close(self):
        self.db.close()

    def reset(self, conversation="default"):
        self.db.execute("DELETE FROM messages WHERE conversation = ?", (conversation,))
        self.db.commit()

    def _next_turn(self, conversation):
        row = self.db.execute(
            "SELECT COALESCE(MAX(turn), 0) + 1 FROM messages WHERE conversation = ?",
            (conversation,),
        ).fetchone()
        return row[0]

    def _history(self, conversation):
        turns = [row[0] for row in self.db.execute(
            "SELECT DISTINCT turn FROM messages WHERE conversation = ? "
            "ORDER BY turn DESC LIMIT ?", (conversation, self.history_turns))]
        if not turns:
            return []
        first = min(turns)
        rows = self.db.execute(
            "SELECT body FROM messages WHERE conversation = ? AND turn >= ? "
            "ORDER BY turn, seq", (conversation, first))
        return [json.loads(row[0]) for row in rows]

    def _save(self, conversation, turn, sequence, message):
        self.db.execute(
            "INSERT INTO messages(conversation, turn, seq, body) VALUES (?, ?, ?, ?)",
            (conversation, turn, sequence, json.dumps(message, ensure_ascii=False)))
        self.db.commit()

    def _tool_result(self, call):
        function = call.get("function") or {}
        name = function.get("name", "")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if not isinstance(arguments, dict):
            result = "Tool error: arguments must be a JSON object."
        else:
            result = self.tools.call(name, arguments)
        response = {"role": "tool", "name": name, "content": result}
        if call.get("id"):
            response["tool_call_id"] = call["id"]
        return response

    def run(self, text, conversation="default"):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("message cannot be empty")
        turn = self._next_turn(conversation)
        user = {"role": "user", "content": text.strip()}
        self._save(conversation, turn, 0, user)
        sequence = 1

        for round_number in range(self.max_tool_rounds + 1):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(self._history(conversation))
            reply = self.client.chat(messages, TOOL_SPECS)
            assistant = {
                "role": "assistant",
                "content": reply.get("content") or "",
            }
            calls = reply.get("tool_calls") or []
            if calls:
                assistant["tool_calls"] = calls
            self._save(conversation, turn, sequence, assistant)
            sequence += 1
            if not calls:
                return assistant["content"].strip()

            # Bound both the number of rounds and the work a single reply can request.
            for call in calls[:8]:
                if round_number == self.max_tool_rounds:
                    result = {"role": "tool", "name": (call.get("function") or {}).get("name", ""),
                              "content": "Tool error: reached the agent tool-round limit."}
                else:
                    result = self._tool_result(call)
                self._save(conversation, turn, sequence, result)
                sequence += 1
        return "Stopped after reaching the tool-call limit."
