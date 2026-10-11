"""Security regression tests for the opt-in read-only workspace tool."""
import tempfile
import unittest
from pathlib import Path

from minagi.workspace_reader import WorkspaceReader


class WorkspaceReaderTests(unittest.TestCase):
    def test_read_workspace_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "repo"
            root.mkdir()
            (root / "a.txt").write_text("hello", encoding="utf-8")
            self.assertEqual(WorkspaceReader(root).read_text("a.txt"), "hello")

    def test_rejects_parent_traversal_and_absolute_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "repo"
            root.mkdir()
            reader = WorkspaceReader(root)
            for path in ("../secret", str(Path(d) / "secret")):
                with self.subTest(path=path), self.assertRaises(PermissionError):
                    reader.read_text(path)

    def test_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "repo"
            root.mkdir()
            secret = Path(d) / "secret"
            secret.write_text("no", encoding="utf-8")
            (root / "link").symlink_to(secret)
            with self.assertRaises(PermissionError):
                WorkspaceReader(root).read_text("link")

    def test_rejects_oversized_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "repo"
            root.mkdir()
            (root / "big.txt").write_text("abcdefgh", encoding="utf-8")
            with self.assertRaises(ValueError):
                WorkspaceReader(root, max_bytes=4).read_text("big.txt")


if __name__ == "__main__":
    unittest.main()
