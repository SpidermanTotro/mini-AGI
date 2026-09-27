import json
import tempfile
import unittest
from pathlib import Path

from minagi.verify_model import inspect_model_artifact


class ModelArtifactVerifierTests(unittest.TestCase):
    def make_model(self, root):
        root = Path(root)
        (root / "experts").mkdir()
        (root / "core.npz").write_bytes(b"core")
        (root / "routers.npz").write_bytes(b"routers")
        (root / "experts" / "e00000.npz").write_bytes(b"expert")
        manifest = {
            "step": 12,
            "val": 1.25,
            "read_chars": 4096,
            "n_experts": 1,
            "d_model": 768,
            "d_ff": 3072,
            "paged": True,
            "total_bytes": 123,
            "experts": [{"id": 0, "file": "e00000.npz"}],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_accepts_complete_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_model(tmp)
            report = inspect_model_artifact(tmp)
        self.assertEqual(report["step"], 12)
        self.assertEqual(report["read_chars"], 4096)
        self.assertEqual(report["n_experts"], 1)
        self.assertTrue(report["paged"])

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


if __name__ == "__main__":
    unittest.main()
