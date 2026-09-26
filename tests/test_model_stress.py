import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from minagi.decode import contrastive_generate
from minagi.report import render_report, weight_stats
from minagi.recur import RecurCoder, RecurConfig
from minagi.stream import ramp_context


def tiny_config(**overrides):
    values = {
        "vocab_size": 37,
        "d_model": 24,
        "n_head": 4,
        "d_ff": 40,
        "block": 12,
        "n_prelude": 1,
        "n_recur": 1,
        "n_coda": 0,
        "max_steps": 3,
        "min_steps": 1,
        "bptt_window": 2,
    }
    values.update(overrides)
    return RecurConfig(**values)


class ModelStressTests(unittest.TestCase):
    def test_context_ramp_reaches_unaligned_endpoint(self):
        self.assertEqual(
            ramp_context(step=100, total=100, start=512, end=800,
                         granularity=256),
            800)

    def test_growth_held_report_uses_converted_step_cadence(self):
        from train import _growth_held_due

        self.assertTrue(_growth_held_due(9_770, 977))
        self.assertFalse(_growth_held_due(20_000_000, 977))

    def test_rejects_invalid_transformer_and_recurrence_configs(self):
        for overrides in (
            {"d_model": 10, "n_head": 3},
            {"d_model": 15, "n_head": 3},
            {"block": 0},
            {"max_steps": 0},
            {"min_steps": 4, "max_steps": 3},
            {"halt_prior": 0.0},
            {"halt_thresh": 1.1},
            {"bptt_window": -1},
            {"pool_experts": 4, "pool_top_k": 5},
            {"pool_experts": 4, "pool_max": 3},
            {"pool_depth": 0},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    tiny_config(**overrides)

    def test_chunked_cache_matches_full_context(self):
        torch.manual_seed(11)
        model = RecurCoder(tiny_config()).eval()
        tokens = torch.randint(model.cfg.vocab_size, (2, 9))
        with torch.no_grad():
            expected = model(tokens)[0]
            caches = model.empty_caches()
            chunks = [
                model(tokens[:, start:start + 3], caches=caches,
                      pos_offset=start)[0]
                for start in range(0, tokens.shape[1], 3)
            ]
        torch.testing.assert_close(torch.cat(chunks, dim=1), expected,
                                   atol=1e-5, rtol=1e-5)

    def test_sampled_depth_stays_fixed_for_live_cache(self):
        model = RecurCoder(tiny_config(max_steps=4, train_steps_mean=1.0)).train()
        samples = iter((4, 1))
        model.sample_depth = lambda: next(samples)
        caches = model.empty_caches()
        for start in (0, 2):
            tokens = torch.randint(model.cfg.vocab_size, (1, 2))
            model(tokens, caches=caches, pos_offset=start)
        self.assertEqual(caches[0]["_depth"], 4)
        self.assertEqual({cache["k"].shape[-2] for cache in caches}, {4})

    def test_training_gradients_and_contrastive_decode(self):
        model = RecurCoder(tiny_config())
        tokens = torch.randint(model.cfg.vocab_size, (2, 6))
        _, loss = model(tokens[:, :-1], targets=tokens[:, 1:])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(
            param.grad is None or torch.isfinite(param.grad).all()
            for param in model.parameters()))

        generated = contrastive_generate(model, tokens[:1, :3], 2, top_k=2)
        self.assertEqual(tuple(generated.shape), (1, 5))
        for arguments in ((-1, 2, 0.5), (1, 0, 0.5), (1, 2, 1.5)):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    contrastive_generate(
                        model, tokens[:1, :3], arguments[0],
                        top_k=arguments[1], alpha=arguments[2])

    def test_weight_report_uses_recurrent_block_layout(self):
        model = RecurCoder(tiny_config(n_prelude=1, n_recur=1, n_coda=1,
                                       max_steps=2))
        stats = weight_stats(model)
        self.assertEqual(stats["n_layer"], 5)
        self.assertEqual(stats["n_block"], 3)
        self.assertIn("L5", render_report(stats))

    def test_shared_pool_forward_and_backward(self):
        model = RecurCoder(tiny_config(
            n_prelude=0, use_pool=True, pool_experts=4, pool_max=4,
            pool_d_ff=8, pool_top_k=2))
        tokens = torch.randint(model.cfg.vocab_size, (2, 5))
        _, loss = model(tokens[:, :-1], targets=tokens[:, 1:])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.pool.gate.grad).all())

    def test_paged_checkpoint_restores_expert_weights_and_adam_moments(self):
        import train
        from minagi.create import create
        from minagi.store import load, save

        with tempfile.TemporaryDirectory() as root, patch.dict(
                "os.environ", {"MINI_AGI_CONFIG": ""}):
            weights = Path(root) / "weights"
            settings = create(
                str(weights), seed=7, verbose=False, d_model=8, n_head=2,
                trunk_d_ff=12, block=8, max_steps=2, experts=4,
                resident=2, d_ff=16, depth=1, top_k=2)
            model, _, pool, _ = train.build_paged(
                str(weights), torch.device("cpu"), resident=2, ram_capacity=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            pool.attach_optimiser(optimizer)
            pool.swap_to([0, 1])

            loss = sum(parameter.square().mean()
                       for parameter in model.parameters())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            expected = {}
            for slot, expert_id in enumerate(pool.slots):
                expected[expert_id] = {}
                for name in ("w1", "w3", "w2"):
                    parameter = getattr(pool, name)
                    state = optimizer.state[parameter]
                    expected[expert_id][name] = parameter[slot].detach().clone()
                    expected[expert_id][name + "_m"] = state["exp_avg"][slot].clone()
                    expected[expert_id][name + "_v"] = state["exp_avg_sq"][slot].clone()
            trunk_parameter = model.tok_emb.weight
            expected_trunk_m = optimizer.state[trunk_parameter]["exp_avg"].clone()

            pool.swap_to([2, 3])
            save(model, str(weights), step=1, opt=optimizer, cfg=settings)

            resumed, _, resumed_pool, _ = train.build_paged(
                str(weights), torch.device("cpu"), resident=2, ram_capacity=4)
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-2)
            resumed_pool.attach_optimiser(resumed_optimizer)
            load(resumed, str(weights), opt=resumed_optimizer,
                 device=torch.device("cpu"))
            resumed_pool.swap_to([0, 1])

            for slot, expert_id in enumerate(resumed_pool.slots):
                for name in ("w1", "w3", "w2"):
                    parameter = getattr(resumed_pool, name)
                    state = resumed_optimizer.state[parameter]
                    torch.testing.assert_close(
                        parameter[slot], expected[expert_id][name])
                    torch.testing.assert_close(
                        state["exp_avg"][slot], expected[expert_id][name + "_m"],
                        atol=0.01, rtol=0.02)
                    torch.testing.assert_close(
                        state["exp_avg_sq"][slot], expected[expert_id][name + "_v"],
                        atol=0.01, rtol=0.02)
            torch.testing.assert_close(
                resumed_optimizer.state[resumed.tok_emb.weight]["exp_avg"],
                expected_trunk_m, atol=0.01, rtol=0.02)

    def test_paged_checkpoint_rejects_missing_expert_file(self):
        import train
        from minagi.create import create
        from minagi.store import save

        with tempfile.TemporaryDirectory() as root, patch.dict(
                "os.environ", {"MINI_AGI_CONFIG": "",
                               "GREENLIGHT_CONFIG": ""}):
            weights = Path(root) / "weights"
            settings = create(
                str(weights), seed=9, verbose=False, d_model=8, n_head=2,
                trunk_d_ff=12, block=8, max_steps=2, experts=4,
                resident=1, d_ff=16, depth=1, top_k=2)
            model, _, pool, manifest = train.build_paged(
                str(weights), torch.device("cpu"), resident=1, ram_capacity=4)
            pool.swap_to([0])
            missing_entry = next(entry for entry in manifest["experts"]
                                 if entry["id"] not in pool.slots)
            (weights / "experts" / missing_entry["file"]).unlink()
            manifest_path = weights / "manifest.json"
            core_path = weights / "core.npz"
            old_manifest = manifest_path.read_bytes()
            old_core = core_path.read_bytes()

            with self.assertRaisesRegex(FileNotFoundError, "missing expert"):
                save(model, str(weights), step=1, cfg=settings)
            self.assertEqual(manifest_path.read_bytes(), old_manifest)
            self.assertEqual(core_path.read_bytes(), old_core)
            with self.assertRaisesRegex(FileNotFoundError, "missing expert"):
                train.build_paged(str(weights), torch.device("cpu"),
                                  resident=1, ram_capacity=4)

    def test_paged_growth_and_pruning_keep_router_rows_and_expert_files_aligned(self):
        import torch.nn as nn
        from minagi.paged import PagedPool
        from minagi.pool import PooledMLP

        with tempfile.TemporaryDirectory() as root:
            pool = PagedPool(root, d_model=4, d_ff=8, n_experts=3,
                             resident=1, ram_capacity=4, device="cpu",
                             max_experts=3)
            for expert_id in range(3):
                value = float(expert_id + 1)
                pool.tiers.put(expert_id, {
                    "w1": torch.full((8, 4), value),
                    "w3": torch.full((8, 4), value),
                    "w2": torch.full((4, 8), value),
                })
            pool.tiers.flush()
            pool.swap_to([0])
            model = nn.Module()
            model.pool = pool
            model.mlp = PooledMLP(pool, d_model=4, top_k=1)
            pool.attach_sites(model)

            def make(expert_id, _source):
                value = float(expert_id + 10)
                return (torch.full((8, 4), value),
                        torch.full((8, 4), value),
                        torch.full((4, 8), value))

            pool.add_experts(1, step=0, make=make)
            self.assertEqual(pool.n_experts(), 4)
            self.assertEqual(pool.segment_router.weight.shape[0], 4)
            self.assertEqual(model.mlp.router.weight.shape[0], 4)
            new_expert_file = Path(root) / "e00003.npz"
            self.assertTrue(new_expert_file.exists())

            self.assertEqual(pool.prune(step=100, survival=1), 3)
            self.assertEqual(pool.n_experts(), 1)
            self.assertEqual(pool.slots, [0])
            self.assertEqual(pool.segment_router.weight.shape[0], 1)
            self.assertEqual(model.mlp.router.weight.shape[0], 1)
            self.assertTrue((Path(root) / "e00000.npz").exists())
            self.assertFalse(new_expert_file.exists())

    def test_invalid_forward_inputs_fail_at_boundary(self):
        model = RecurCoder(tiny_config())
        tokens = torch.ones((1, 2), dtype=torch.long)
        calls = (
            lambda: model(tokens[0]),
            lambda: model(tokens[:, :0]),
            lambda: model(tokens, pos_offset=-1),
            lambda: model(tokens, caches=[]),
            lambda: model(tokens, targets=tokens[:, :1]),
            lambda: model(tokens, targets=tokens.to(torch.int32)),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
