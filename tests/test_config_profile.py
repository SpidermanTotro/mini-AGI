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
        with patch.dict(os.environ, {
            "GREENLIGHT_CONFIG": str(PROFILE), "MINI_AGI_CONFIG": ""}):
            settings = load()

        self.assertEqual(get(settings, "model.d_model"), 768)
        self.assertEqual(get(settings, "model.n_head"), 12)
        self.assertEqual(get(settings, "model.d_ff"), 2048)
        self.assertEqual(get(settings, "pool.experts"), 128)
        self.assertEqual(get(settings, "pool.width"), 3072)
        self.assertEqual(get(settings, "pool.resident"), 64)
        self.assertEqual(get(settings, "pool.ram_cache"), 128)
        self.assertEqual(get(settings, "model.context_end"), 4096)
        self.assertEqual(get(settings, "data.weights"), "greenlight-16g-r1")
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
        expert_params = 3 * cfg.d_model * cfg.pool_d_ff
        resident_state_gib = (
            get(settings, "pool.resident") * expert_params * 16 / 1024**3)
        ram_cache_gib = (
            get(settings, "pool.ram_cache") * expert_params * 12 / 1024**3)
        self.assertLess(resident_state_gib, 8.0)
        self.assertLess(ram_cache_gib, 11.0)

    def test_default_profile_remains_unchanged_without_override(self):
        with patch.dict(os.environ, {
            "GREENLIGHT_CONFIG": "", "MINI_AGI_CONFIG": ""}):
            settings = load()
        self.assertEqual(get(settings, "model.d_model"), 512)
        self.assertEqual(get(settings, "pool.resident"), 32)
        self.assertEqual(get(settings, "model.context_end"), 8192)

    def test_default_router_balance_is_off(self):
        with patch.dict(os.environ, {
            "GREENLIGHT_CONFIG": "", "MINI_AGI_CONFIG": ""}):
            settings = load()
        self.assertEqual(get(settings, "pool.balance"), 0.0)
        self.assertGreater(get(settings, "pool.explore_bias"), 0.0)

    def test_legacy_config_environment_variable_remains_supported(self):
        with patch.dict(os.environ, {
                "GREENLIGHT_CONFIG": "", "MINI_AGI_CONFIG": str(PROFILE)}):
            settings = load()
        self.assertEqual(get(settings, "data.weights"), "greenlight-16g-r1")

    def test_greenlight_config_takes_precedence_over_legacy_variable(self):
        with patch.dict(os.environ, {
                "GREENLIGHT_CONFIG": str(PROFILE),
                "MINI_AGI_CONFIG": "/does/not/exist.yaml"}):
            settings = load()
        self.assertEqual(get(settings, "data.weights"), "greenlight-16g-r1")


if __name__ == "__main__":
    unittest.main()
