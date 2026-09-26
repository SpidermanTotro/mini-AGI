import os
import unittest
from pathlib import Path
from unittest.mock import patch

from minagi.config import get, load
from minagi.recur import RecurConfig


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config-16gb.yaml"


class ConfigProfileTests(unittest.TestCase):
    def test_16gb_profile_is_selectable_and_architecture_is_valid(self):
        with patch.dict(os.environ, {"MINI_AGI_CONFIG": str(PROFILE)}):
            settings = load()

        self.assertEqual(get(settings, "model.d_model"), 512)
        self.assertEqual(get(settings, "pool.width"), 2048)
        self.assertEqual(get(settings, "pool.resident"), 56)
        self.assertEqual(get(settings, "pool.ram_cache"), 192)
        self.assertEqual(get(settings, "model.context_end"), 4096)
        self.assertEqual(get(settings, "data.weights"), "agi-16")
        cfg = RecurConfig(
            vocab_size=265,
            d_model=settings["model"]["d_model"],
            n_head=settings["model"]["n_head"],
            d_ff=settings["model"]["d_ff"],
            block=settings["model"]["context_end"],
            n_prelude=settings["model"]["n_prelude"],
            n_recur=settings["model"]["n_recur"],
            n_coda=settings["model"]["n_coda"],
            max_steps=settings["model"]["max_steps"],
            use_pool=True,
            pool_experts=settings["pool"]["experts"],
            pool_d_ff=settings["pool"]["width"],
            pool_depth=settings["pool"]["depth"],
            pool_top_k=settings["pool"]["top_k"],
            pool_max=settings["pool"]["experts"],
        )
        self.assertEqual(cfg.d_model // cfg.n_head, 64)
        self.assertEqual(cfg.n_layer_effective, 26)

    def test_default_profile_remains_unchanged_without_override(self):
        with patch.dict(os.environ, {"MINI_AGI_CONFIG": ""}):
            settings = load()
        self.assertEqual(get(settings, "model.d_model"), 512)
        self.assertEqual(get(settings, "pool.resident"), 32)
        self.assertEqual(get(settings, "model.context_end"), 8192)


if __name__ == "__main__":
    unittest.main()
