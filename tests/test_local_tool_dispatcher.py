import tempfile
import unittest
from pathlib import Path

from minagi.local_assistant_tools import LocalAssistantTools
from minagi.local_tool_dispatcher import LocalToolDispatcher


class DispatcherTests(unittest.TestCase):
    def test_permission_denied_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example.txt").write_text("safe", encoding="utf-8")
            dispatcher = LocalToolDispatcher(LocalAssistantTools(root, root / "history.db"))
            self.assertEqual(
                dispatcher.dispatch("read_workspace_file", {"relative_path": "example.txt"}),
                {"ok": False, "error": "PermissionError"},
            )

    def test_explicit_permission_reads_only_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example.txt").write_text("safe", encoding="utf-8")
            dispatcher = LocalToolDispatcher(LocalAssistantTools(
                root, root / "history.db", allow_workspace_reads=True))
            self.assertEqual(
                dispatcher.dispatch_json("read_workspace_file", '{"relative_path":"example.txt"}'),
                {"ok": True, "content": "safe"},
            )
            self.assertFalse(dispatcher.dispatch(
                "read_workspace_file", {"relative_path": "../outside.txt"})["ok"])

    def test_unknown_tool_and_bad_arguments_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dispatcher = LocalToolDispatcher(LocalAssistantTools(root, root / "history.db"))
            self.assertEqual(dispatcher.dispatch("run_shell", {"command": "ls"})["error"],
                             "tool not allowed")
            self.assertEqual(dispatcher.dispatch("read_workspace_file", {})["error"],
                             "invalid arguments")
            self.assertEqual(dispatcher.dispatch_json("read_workspace_file", "{")["error"],
                             "invalid JSON")
            self.assertEqual(dispatcher.dispatch_json("read_workspace_file", "x" * 8193)["error"],
                             "arguments too large")


if __name__ == "__main__":
    unittest.main()
