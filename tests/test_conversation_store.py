"""Tests for optional local conversation storage."""
import tempfile
import unittest
from pathlib import Path

from minagi.conversation_store import ConversationStore


class ConversationStoreTests(unittest.TestCase):
    def test_persists_and_isolates_conversations(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "memory.sqlite3"
            with ConversationStore(path) as store:
                store.add("a", "user", "hello")
                store.add("b", "assistant", "other")
                store.add("a", "assistant", "hi")
            with ConversationStore(path) as store:
                self.assertEqual(
                    [(x["role"], x["content"]) for x in store.history("a")],
                    [("user", "hello"), ("assistant", "hi")],
                )
                self.assertEqual(len(store.history("b")), 1)

    def test_deletes_only_requested_conversation(self):
        with tempfile.TemporaryDirectory() as d:
            with ConversationStore(Path(d) / "memory.sqlite3") as store:
                store.add("a", "user", "private")
                store.add("b", "user", "keep")
                self.assertEqual(store.delete_conversation("a"), 1)
                self.assertEqual(store.history("a"), [])
                self.assertEqual(len(store.history("b")), 1)

    def test_invalid_role_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with ConversationStore(Path(d) / "memory.sqlite3") as store:
                with self.assertRaises(ValueError):
                    store.add("a", "unknown", "message")


if __name__ == "__main__":
    unittest.main()
