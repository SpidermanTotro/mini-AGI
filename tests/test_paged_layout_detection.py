import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from minagi.recur import _load_dir


class PagedLayoutDetectionTests(unittest.TestCase):
    def test_experts_directory_selects_paged_loader_without_manifest_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experts").mkdir()
            (root / "manifest.json").write_text(json.dumps({
                "cfg": {},
                "n_experts": 1
            }))

            sentinel_model = object()
            sentinel_cfg = type("Cfg", (), {"__dict__": {"block": 16}})()
            with mock.patch("train.build_paged",
                            return_value=(sentinel_model, sentinel_cfg, object(),
                                          {"step": 7, "val": 1.25})) as build:
                model, meta = _load_dir(str(root), "cpu")

            self.assertIs(model, sentinel_model)
            build.assert_called_once_with(str(root), "cpu", read_only=False)
            self.assertEqual(meta["step"], 7)


if __name__ == "__main__":
    unittest.main()
