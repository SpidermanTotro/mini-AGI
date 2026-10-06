import tempfile
import unittest
from pathlib import Path

import yaml

from tools.prepare_balance_ab import prepare


class PrepareBalanceABTests(unittest.TestCase):
    def test_generates_profiles_that_differ_only_in_routing_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.yaml"
            source.write_text(yaml.safe_dump({
                "model": {"d_model": 768},
                "pool": {"experts": 128, "explore_bias": 0.65,
                         "balance": 0.0, "resident": 64},
                "training": {"seed": 123},
            }, sort_keys=False))
            out = prepare(source, root / "ab", "python train.py stream corpus")
            base = yaml.safe_load(Path(out["baseline"]["config"]).read_text())
            cand = yaml.safe_load(Path(out["candidate"]["config"]).read_text())

            self.assertEqual(base["pool"]["explore_bias"], 0.65)
            self.assertEqual(base["pool"]["balance"], 0.0)
            self.assertEqual(cand["pool"]["explore_bias"], 0.0)
            self.assertEqual(cand["pool"]["balance"], 3.5e-4)

            for cfg in (base, cand):
                cfg["pool"].pop("explore_bias")
                cfg["pool"].pop("balance")
            self.assertEqual(base, cand)
            self.assertEqual(out["baseline"]["command"].split(" ", 1)[1],
                             out["candidate"]["command"].split(" ", 1)[1])

    def test_rejects_config_without_pool_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.yaml"
            source.write_text("model:\n  d_model: 768\n")
            with self.assertRaisesRegex(ValueError, "pool mapping"):
                prepare(source, Path(tmp) / "ab", "python train.py")


if __name__ == "__main__":
    unittest.main()
