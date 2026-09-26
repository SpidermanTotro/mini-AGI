import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class FetchCacheTests(unittest.TestCase):
    def _run_fetch(self, root, keep_cache=False):
        out = Path(root) / "out" / "chat"
        shared = Path(root) / "shared-hf-cache"
        shared.mkdir()
        marker = shared / "keep-me.txt"
        marker.write_text("unrelated")

        seen = {}
        fake = types.ModuleType("datasets")

        def load_dataset(*args, **kwargs):
            seen["cache_dir"] = kwargs.get("cache_dir")
            return [{"conversations": [
                {"from": "user", "value": "hello"},
                {"from": "assistant", "value": "world"},
            ]}]

        fake.load_dataset = load_dataset
        argv = ["corpora.fetch", "--dataset", "fake/data", "--kind", "chat",
                "--out", str(out)]
        if keep_cache:
            argv.append("--keep-cache")

        with mock.patch.dict(sys.modules, {"datasets": fake}), \
             mock.patch.object(sys, "argv", argv):
            from corpora import fetch
            self.assertEqual(fetch.main(), 0)

        self.assertTrue(marker.exists())
        return seen["cache_dir"]

    def test_default_uses_and_removes_private_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._run_fetch(tmp)
            self.assertIsNotNone(cache)
            self.assertFalse(Path(cache).exists())

    def test_keep_cache_uses_hugging_face_default_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._run_fetch(tmp, keep_cache=True)
            self.assertIsNone(cache)


if __name__ == "__main__":
    unittest.main()
