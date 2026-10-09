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
            out = prepare(source, root / "ab", "python train.py stream --weights-dir {weights_dir} --out {run_dir}")
            base = yaml.safe_load(Path(out["baseline"]["config"]).read_text())
            cand = yaml.safe_load(Path(out["candidate"]["config"]).read_text())

            self.assertEqual(base["pool"]["explore_bias"], 0.0)
            self.assertEqual(base["pool"]["balance"], 0.0)
            self.assertEqual(cand["pool"]["explore_bias"], 0.0)
            self.assertEqual(cand["pool"]["balance"], 0.001)

            self.assertEqual(cand["pool"]["explore_bias"], base["pool"]["explore_bias"])
            changed = [(section, key) for section in base
                       for key in base[section] if isinstance(base[section], dict)
                       and base[section][key] != cand[section][key]]
            self.assertEqual(changed, [("pool", "balance")])
            self.assertNotEqual(out["baseline"]["weights_dir"],
                                out["candidate"]["weights_dir"])
            self.assertNotEqual(out["baseline"]["run_dir"],
                                out["candidate"]["run_dir"])
            for arm in ("baseline", "candidate"):
                self.assertIn("--weights-dir " + out[arm]["weights_dir"],
                              out[arm]["command"])
                self.assertIn("--out " + out[arm]["run_dir"],
                              out[arm]["command"])
                self.assertFalse(Path(out[arm]["weights_dir"]).exists())
                self.assertFalse(Path(out[arm]["run_dir"]).exists())

    def test_refuses_shared_default_weights_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.yaml"
            source.write_text("pool:\n  experts: 4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "weights_dir"):
                prepare(source, Path(tmp) / "ab", "python train.py stream")

    def test_never_overwrites_frozen_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.yaml"
            source.write_text("pool:\n  experts: 4\n", encoding="utf-8")
            out = Path(tmp) / "ab"
            prepare(source, out, "python train.py stream --weights-dir {weights_dir} --out {run_dir}")
            before = (out / "baseline.yaml").read_bytes()
            with self.assertRaises(FileExistsError):
                prepare(source, out, "python train.py stream --weights-dir {weights_dir} --out {run_dir}")
            self.assertEqual((out / "baseline.yaml").read_bytes(), before)

    def test_rejects_config_without_pool_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.yaml"
            source.write_text("model:\n  d_model: 768\n")
            with self.assertRaisesRegex(ValueError, "pool mapping"):
                prepare(source, Path(tmp) / "ab", "python train.py")


if __name__ == "__main__":
    unittest.main()
