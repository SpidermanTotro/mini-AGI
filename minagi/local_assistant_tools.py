"""Explicit-permission local assistant tools (no model or network dependency)."""
from __future__ import annotations

from pathlib import Path

from minagi.conversation_store import ConversationStore
from minagi.workspace_reader import WorkspaceReader


class LocalAssistantTools:
    """Host-owned, opt-in tool surface; never grants permissions automatically."""

    def __init__(self, workspace: str | Path, history_db: str | Path,
                 *, allow_workspace_reads: bool = False):
        self.reader = WorkspaceReader(workspace)
        self.history_db = Path(history_db)
        self.allow_workspace_reads = allow_workspace_reads

    def read_workspace_file(self, relative_path: str) -> str:
        if not self.allow_workspace_reads:
            raise PermissionError("workspace reading requires explicit permission")
        return self.reader.read_text(relative_path)

    def save_message(self, conversation_id: str, role: str, content: str) -> int:
        with ConversationStore(self.history_db) as store:
            return store.add(conversation_id, role, content)

    def conversation_history(self, conversation_id: str) -> list[dict]:
        with ConversationStore(self.history_db) as store:
            return store.history(conversation_id)

    def delete_conversation(self, conversation_id: str) -> int:
        with ConversationStore(self.history_db) as store:
            return store.delete_conversation(conversation_id)
