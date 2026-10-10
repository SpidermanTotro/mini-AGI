"""Integration tests for explicit-permission local assistant tools."""
import tempfile
import unittest
from pathlib import Path

from minagi.local_assistant_tools import LocalAssistantTools


class LocalAssistantToolsTests(unittest.TestCase):
    def test_workspace_read_denied_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "file.txt").write_text("hello", encoding="utf-8")
            tools = LocalAssistantTools(root, root / "memory.sqlite3")
            with self.assertRaises(PermissionError):
                tools.read_workspace_file("file.txt")

    def test_explicit_read_permission(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "file.txt").write_text("hello", encoding="utf-8")
            tools = LocalAssistantTools(
                root, root / "memory.sqlite3", allow_workspace_reads=True
            )
            self.assertEqual(tools.read_workspace_file("file.txt"), "hello")

    def test_history_save_read_delete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tools = LocalAssistantTools(root, root / "memory.sqlite3")
            tools.save_message("chat1", "user", "hello")
            self.assertEqual(tools.conversation_history("chat1")[0]["content"], "hello")
            self.assertEqual(tools.delete_conversation("chat1"), 1)
            self.assertEqual(tools.conversation_history("chat1"), [])


if __name__ == "__main__":
    unittest.main()
