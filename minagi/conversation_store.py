"""Opt-in local SQLite conversation storage for Greenlight Next.

This module does not contact a network service or modify model checkpoints.
Callers choose the database path and explicitly open the store.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


class ConversationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "conversation_id TEXT NOT NULL, role TEXT NOT NULL, "
            "content TEXT NOT NULL, created_at TEXT NOT NULL "
            "DEFAULT CURRENT_TIMESTAMP)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation "
            "ON messages(conversation_id, id)"
        )
        self.conn.commit()

    def add(self, conversation_id: str, role: str, content: str) -> int:
        if not conversation_id or role not in ("system", "user", "assistant", "tool"):
            raise ValueError("invalid conversation ID or role")
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content),
            )
        return int(cur.lastrowid)

    def history(self, conversation_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, role, content FROM messages "
            "WHERE conversation_id = ? ORDER BY id", (conversation_id,)
        )
        return [{"id": i, "role": role, "content": content}
                for i, role, content in rows]

    def delete_conversation(self, conversation_id: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
        return cur.rowcount

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ConversationStore":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
