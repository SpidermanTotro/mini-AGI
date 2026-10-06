import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minagi.verify_model import inspect_model_artifact


class ModelArtifactVerifierTests(unittest.TestCase):
    def make_model(self, root):
        root = Path(root)
        (root / "experts").mkdir()
        np.savez(root / "core.npz", embedding=np.zeros((3, 4)), norm=np.zeros(4))
        np.savez(root / "routers.npz", router=np.zeros((2, 4)))
        np.savez(root / "experts" / "e00000.npz", w1=np.zeros((5, 4)))
        manifest = {
            "step": 12, "val": 1.25, "read_chars": 4096,
            "n_experts": 1, "d_model": 4, "d_ff": 5, "paged": True,
            "total_bytes": 123, "cfg": {"pool_resident": 1},
            "experts": [{"id": 0, "file": "e00000.npz", "params": 20}],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_accepts_complete_model_directory_and_counts_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_model(tmp)
            report = inspect_model_artifact(tmp)
        self.assertEqual(report["step"], 12)
        self.assertEqual(report["read_chars"], 4096)
        self.assertEqual(report["n_experts"], 1)
        self.assertTrue(report["paged"])
        self.assertEqual(report["params"]["core"], 16)
        self.assertEqual(report["params"]["routers"], 8)
        self.assertEqual(report["params"]["experts"], 20)
        self.assertEqual(report["params"]["total"], 44)
        self.assertEqual(report["params"]["resident"], 44)

    def test_rejects_missing_core_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_model(tmp)
            (Path(tmp) / "core.npz").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "core.npz"):
                inspect_model_artifact(tmp)

    def test_rejects_missing_expert_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_model(tmp)
            (Path(tmp) / "experts" / "e00000.npz").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "e00000.npz"):
                inspect_model_artifact(tmp)

    def test_rejects_manifest_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_model(tmp)
            path = Path(tmp) / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["n_experts"] = 2
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "declares 2 experts"):
                inspect_model_artifact(tmp)

    def test_rejects_missing_expert_parameter_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_model(tmp)
            path = Path(tmp) / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            del manifest["experts"][0]["params"]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no parameter count"):
                inspect_model_artifact(tmp)


if __name__ == "__main__":
    unittest.main()
